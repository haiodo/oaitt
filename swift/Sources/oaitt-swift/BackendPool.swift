// Pool of transcription backends behind the balancer.
//
// A backend is just a URL, so a local Swift worker, the Python service on 9007 and any
// other OpenAI-compatible endpoint all go into the same pool. Routing is
// least-outstanding rather than round-robin: transcription times differ by an order of
// magnitude between a 2-second reply and a 30-minute recording, and round-robin would
// happily queue a short request behind a long one.

import Foundation

public final class BackendPool: @unchecked Sendable {
    public struct Status: Sendable {
        public let url: String
        public let healthy: Bool
        public let inFlight: Int
        public let completed: Int
        public let failures: Int
        public let lastError: String?
    }

    private final class Backend {
        let url: URL
        var healthy = true
        var inFlight = 0
        var completed = 0
        var failures = 0
        /// Подряд идущие ошибки: один таймаут - ещё не повод выбить бэкенд из ротации,
        /// а вот три подряд - уже повод.
        var consecutiveFailures = 0
        var lastSuccess = Date()
        var lastError: String?

        init(url: URL) { self.url = url }
    }

    /// Сколько ошибок подряд терпим до вывода из ротации.
    private static let failureThreshold = 3
    /// Через сколько без единого успешного ответа считаем бэкенд мёртвым, даже если у
    /// него висят запросы: иначе зависший бэкенд остаётся здоровым навсегда.
    private static let stallTimeout: TimeInterval = 120

    private var backends: [Backend]
    private let lock = NSLock()
    private let session: URLSession

    public init(urls: [URL], requestTimeout: TimeInterval) {
        self.backends = urls.map(Backend.init)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = requestTimeout
        configuration.timeoutIntervalForResource = requestTimeout
        configuration.httpMaximumConnectionsPerHost = 32
        self.session = URLSession(configuration: configuration)
    }

    public var count: Int { backends.count }

    public var statuses: [Status] {
        lock.withLock {
            backends.map {
                Status(
                    url: $0.url.absoluteString, healthy: $0.healthy, inFlight: $0.inFlight,
                    completed: $0.completed, failures: $0.failures, lastError: $0.lastError)
            }
        }
    }

    /// Least-outstanding healthy backend, skipping ones already tried for this request.
    private func pick(excluding tried: Set<String>) -> Backend? {
        lock.withLock {
            let candidates = backends.filter {
                $0.healthy && !tried.contains($0.url.absoluteString)
            }
            // All unhealthy is not a reason to drop the request on the floor: health checks
            // can lag behind a backend that has just come back.
            let pool =
                candidates.isEmpty
                ? backends.filter { !tried.contains($0.url.absoluteString) }
                : candidates
            guard let chosen = pool.min(by: { $0.inFlight < $1.inFlight }) else { return nil }
            chosen.inFlight += 1
            return chosen
        }
    }

    private func finish(_ backend: Backend, error: Error?) {
        lock.withLock {
            backend.inFlight -= 1
            if let error {
                backend.failures += 1
                backend.consecutiveFailures += 1
                backend.lastError = String(describing: error)
                if backend.consecutiveFailures >= Self.failureThreshold {
                    backend.healthy = false
                }
            } else {
                backend.completed += 1
                backend.consecutiveFailures = 0
                backend.lastSuccess = Date()
                backend.healthy = true
                backend.lastError = nil
            }
        }
    }

    public struct ProxyResponse: Sendable {
        public let status: Int
        public let contentType: String
        public let body: Data
    }

    /// Forwards the request body untouched. The balancer never parses multipart - it has
    /// no reason to know what is inside, and re-encoding a 4 MB upload would cost more
    /// than the routing decision.
    public func forward(
        path: String, contentType: String, body: Data, authorization: String?, attempts: Int = 2
    ) async throws -> ProxyResponse {
        var tried: Set<String> = []
        var lastError: Error = BalancerError.noBackends

        for _ in 0..<max(1, attempts) {
            guard let backend = pick(excluding: tried) else { break }
            tried.insert(backend.url.absoluteString)

            var request = URLRequest(url: backend.url.appendingPathComponent(path))
            request.httpMethod = "POST"
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
            if let authorization {
                request.setValue(authorization, forHTTPHeaderField: "Authorization")
            }
            request.httpBody = body

            do {
                let (data, response) = try await session.data(for: request)
                let http = response as? HTTPURLResponse
                let status = http?.statusCode ?? 502
                // 5xx means this backend is sick; 4xx is the caller's problem and retrying
                // it elsewhere would only duplicate the error.
                if status >= 500 {
                    finish(backend, error: BalancerError.badStatus(status))
                    lastError = BalancerError.badStatus(status)
                    continue
                }
                finish(backend, error: nil)
                return ProxyResponse(
                    status: status,
                    contentType: http?.value(forHTTPHeaderField: "Content-Type")
                        ?? "application/json",
                    body: data)
            } catch {
                finish(backend, error: error)
                lastError = error
            }
        }
        throw lastError
    }

    /// Marks backends up or down by polling /health.
    public func runHealthChecks(every interval: TimeInterval) async {
        while !Task.isCancelled {
            for backend in lock.withLock({ backends }) {
                var request = URLRequest(url: backend.url.appendingPathComponent("health"))
                request.timeoutInterval = min(5, interval)
                let healthy: Bool
                do {
                    let (_, response) = try await session.data(for: request)
                    healthy = (response as? HTTPURLResponse)?.statusCode == 200
                } catch {
                    healthy = false
                }
                lock.withLock {
                    if healthy {
                        backend.healthy = true
                        backend.consecutiveFailures = 0
                        backend.lastSuccess = Date()
                    } else if backend.inFlight == 0
                        || Date().timeIntervalSince(backend.lastSuccess) > Self.stallTimeout
                    {
                        // Запросы в полёте раньше защищали бэкенд от пометки больным, но
                        // зависший так и оставался здоровым: он "занят" и не отвечает.
                        backend.healthy = false
                    }
                }
            }
            try? await Task.sleep(for: .seconds(interval))
        }
    }
}

public enum BalancerError: Error, CustomStringConvertible {
    case noBackends
    case badStatus(Int)
    case queueFull

    public var description: String {
        switch self {
        case .noBackends: return "no backend available"
        case .badStatus(let code): return "backend returned \(code)"
        case .queueFull: return "balancer queue is full"
        }
    }
}
