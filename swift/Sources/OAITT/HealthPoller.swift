import Foundation
import Observation

/// Reads the balancer's own /health/detailed - it already knows which backends are up
/// and how loaded they are, so the app does not poll workers itself.
@MainActor
@Observable
final class HealthPoller {
    private(set) var healthyBackends = 0
    private(set) var inFlight = 0
    private(set) var completed = 0
    private(set) var reachable = false

    private var task: Task<Void, Never>?

    func start(port: Int, apiKey: String, interval: TimeInterval = 2) {
        stop()
        task = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll(port: port, apiKey: apiKey)
                try? await Task.sleep(for: .seconds(interval))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        healthyBackends = 0
        inFlight = 0
        reachable = false
    }

    private func poll(port: Int, apiKey: String) async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health/detailed") else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        guard let (data, _) = try? await URLSession.shared.data(for: request),
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            reachable = false
            return
        }

        reachable = true
        inFlight = payload["in_flight"] as? Int ?? 0

        // С одним воркером балансировщика нет, и на порту отвечает сам воркер - у него
        // другой ответ: не список бэкендов, а собственная модель и телеметрия.
        if let backends = payload["backends"] as? [[String: Any]] {
            healthyBackends = backends.filter { $0["healthy"] as? Bool == true }.count
            completed = backends.reduce(0) { $0 + ($1["completed"] as? Int ?? 0) }
        } else {
            healthyBackends = 1
            let telemetry = payload["telemetry"] as? [String: Any] ?? [:]
            completed = telemetry["requests"] as? Int ?? 0
        }
    }
}
