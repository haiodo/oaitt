import Darwin
import Foundation
import Observation

/// Per-worker liveness and cost.
///
/// Two sources, because neither alone is enough: /health/detailed knows whether the model
/// is loaded and how much unified memory it holds, and `ps` knows CPU and resident size of
/// the process itself. A worker answering HTTP but pinning a core looks fine in one and
/// wrong in the other.
@MainActor
@Observable
final class WorkerMonitor {
    struct Row: Identifiable {
        let id: Int
        let port: Int
        let pid: Int32
        var responding = false
        var modelLoaded = false
        var idleTimeout: Double = 0
        var gpuMB: Double = 0
        /// phys_footprint - то же число, что Activity Monitor показывает в колонке Memory.
        /// RSS для этих процессов почти ничего не говорит: веса живут в unified memory,
        /// и `ps -o rss` показывал 26 МБ там, где реально занято полтора гигабайта.
        var footprintMB: Double = 0
        var cpuPercent: Double = 0
        var uptime: TimeInterval = 0
        var restarts = 0
    }

    private(set) var rows: [Row] = []
    private var task: Task<Void, Never>?
    private var samples: [Int32: (cpuNanos: UInt64, at: Date)] = [:]

    func start(supervisor: Supervisor, interval: TimeInterval = 3) {
        stop()
        task = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll(supervisor: supervisor)
                try? await Task.sleep(for: .seconds(interval))
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        rows = []
    }

    private func poll(supervisor: Supervisor) async {
        let workers = supervisor.workers
        let usage = processUsage(pids: workers.map(\.pid))
        var updated: [Row] = []

        for worker in workers {
            var row = Row(id: worker.id, port: worker.port, pid: worker.pid)
            row.restarts = worker.restarts
            row.uptime = Date().timeIntervalSince(worker.startedAt)
            if let usage = usage[worker.pid] {
                row.cpuPercent = usage.cpu
                row.footprintMB = usage.footprintMB
            }
            if let health = await Self.health(port: worker.port) {
                row.responding = true
                let model = health["model"] as? [String: Any] ?? [:]
                row.modelLoaded = model["loaded"] as? Bool ?? false
                row.idleTimeout = model["idle_timeout"] as? Double ?? 0
                let memory = health["memory"] as? [String: Any] ?? [:]
                row.gpuMB = memory["gpu_memory_mb"] as? Double ?? 0
            }
            updated.append(row)
        }
        rows = updated
    }

    private static func health(port: Int) async -> [String: Any]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health/detailed") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    /// CPU считается по дельте потреблённого времени между опросами: rusage отдаёт
    /// суммарное время с запуска, а показывать надо текущую нагрузку.
    private func processUsage(pids: [Int32]) -> [Int32: (cpu: Double, footprintMB: Double)] {
        let now = Date()
        var result: [Int32: (cpu: Double, footprintMB: Double)] = [:]
        // pid переиспользуются системой, а воркеры перезапускаются - держать замеры
        // умерших процессов значит однажды приписать их CPU чужому.
        samples = samples.filter { pids.contains($0.key) }

        for pid in pids {
            var info = rusage_info_v4()
            let status = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                    proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
                }
            }
            guard status == 0 else { continue }

            let cpuNanos = info.ri_user_time + info.ri_system_time
            var percent = 0.0
            if let previous = samples[pid] {
                let elapsed = now.timeIntervalSince(previous.at)
                if elapsed > 0 {
                    percent = Double(cpuNanos &- previous.cpuNanos) / 1_000_000_000 / elapsed * 100
                }
            }
            samples[pid] = (cpuNanos, now)
            result[pid] = (max(0, percent), Double(info.ri_phys_footprint) / 1024 / 1024)
        }
        return result
    }
}
