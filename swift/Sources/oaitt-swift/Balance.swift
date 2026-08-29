import ArgumentParser
import Foundation
import GigaAM
import Hummingbird

/// OpenAI-compatible front end over a pool of transcription backends.
///
/// A single Swift worker does not scale inside its process, so throughput comes from
/// running several and spreading requests across them. Backends are plain URLs, so the
/// Python service can sit in the same pool.
struct Balance: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Spread OpenAI transcription requests across several backends.")

    @Option var host = "127.0.0.1"
    @Option var port = 9007

    @Option(help: "Backend base URLs, comma separated.")
    var backends: String

    /// Same default as the Python service (AUTH_TOKEN=key); empty disables auth.
    @Option(help: "Bearer token clients must send; empty disables auth.")
    var apiKey = "key"

    @Option(help: "Bearer token sent to backends; defaults to the client's own.")
    var backendKey: String?

    @Option(help: "Seconds between /health polls of each backend.")
    var healthInterval = 5.0

    @Option(help: "Seconds a single proxied request may take.")
    var requestTimeout = 600.0

    @Option(help: "Requests allowed in flight before the balancer answers 503.")
    var maxInFlight = 64

    func run() async throws {
        let urls = backends.split(separator: ",").compactMap {
            URL(string: $0.trimmingCharacters(in: .whitespaces))
        }
        guard !urls.isEmpty else {
            throw ValidationError("no valid backend URLs in --backends")
        }

        let pool = BackendPool(urls: urls, requestTimeout: requestTimeout)
        let inFlight = InFlightCounter(limit: maxInFlight)
        let token = apiKey
        let backendToken = backendKey
        let router = Router()

        router.get("/health") { _, _ -> Response in
            let statuses = pool.statuses
            return Self.json([
                "status": statuses.contains(where: \.healthy) ? "healthy" : "degraded",
                "engine": "oaitt-balancer",
                "backends": statuses.count,
                "healthy": statuses.filter(\.healthy).count,
            ])
        }

        router.get("/health/detailed") { _, _ -> Response in
            Self.json([
                "status": "healthy",
                "engine": "oaitt-balancer",
                "in_flight": inFlight.current,
                "max_in_flight": maxInFlight,
                "backends": pool.statuses.map { status in
                    var row: [String: Any] = [
                        "url": status.url, "healthy": status.healthy,
                        "in_flight": status.inFlight, "completed": status.completed,
                        "failures": status.failures,
                    ]
                    if let error = status.lastError { row["last_error"] = error }
                    return row
                },
            ])
        }

        for path in ["/v1/audio/transcriptions", "/asr"] {
            router.post(RouterPath(path)) { request, _ -> Response in
                if !token.isEmpty, request.headers[.authorization] != "Bearer \(token)" {
                    throw HTTPError(.unauthorized)
                }
                guard inFlight.enter() else {
                    throw HTTPError(.serviceUnavailable, message: "balancer queue is full")
                }
                defer { inFlight.leave() }

                let buffer = try await request.body.collect(upTo: 512 * 1024 * 1024)
                let authorization =
                    backendToken.map { "Bearer \($0)" }
                    ?? request.headers[.authorization]

                do {
                    let proxied = try await pool.forward(
                        path: String(path.dropFirst()),
                        contentType: request.headers[.contentType] ?? "application/octet-stream",
                        body: Data(buffer.readableBytesView),
                        authorization: authorization)
                    return Response(
                        status: .init(code: proxied.status),
                        headers: [.contentType: proxied.contentType],
                        body: .init(byteBuffer: ByteBuffer(bytes: proxied.body)))
                } catch {
                    throw HTTPError(.badGateway, message: String(describing: error))
                }
            }
        }

        let app = Application(
            router: router,
            server: .http1(
                configuration: .init(additionalChannelHandlers: [ExpectContinueHandler()])),
            configuration: .init(address: .hostname(host, port: port), serverName: "oaitt-balancer")
        )

        let health = Task { await pool.runHealthChecks(every: healthInterval) }
        defer { health.cancel() }
        try await app.runService()
    }

    static func json(_ payload: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return Response(
            status: .ok, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data)))
    }
}

/// Bounds work in flight so a burst queues in the client rather than in memory here.
final class InFlightCounter: @unchecked Sendable {
    private var value = 0
    private let limit: Int
    private let lock = NSLock()

    init(limit: Int) { self.limit = limit }

    var current: Int { lock.withLock { value } }

    func enter() -> Bool {
        lock.withLock {
            guard value < limit else { return false }
            value += 1
            return true
        }
    }

    func leave() {
        lock.withLock { value -= 1 }
    }
}
