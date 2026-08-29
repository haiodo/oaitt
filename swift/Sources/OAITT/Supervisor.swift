// Runs the worker processes and the balancer, and keeps them alive.
//
// Workers are separate processes on purpose: a single Swift process stops scaling at
// concurrency 1 - MLX serialises inside it, and a pool of model copies in one process
// changes nothing (143x against 144x for one). Throughput comes from more processes:
// 144 -> 263 -> 369x for 1, 2 and 4.

import Foundation
import Observation

@Observable
final class Supervisor: @unchecked Sendable {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case failed(String)
    }

    struct WorkerInfo: Identifiable, Equatable {
        let id: Int
        let port: Int
        var pid: Int32
        var alive: Bool
        var restarts: Int
        var startedAt: Date
    }

    private(set) var state: State = .stopped
    private(set) var workers: [WorkerInfo] = []
    private(set) var balancerPort: Int = 0

    /// Воркер, падающий сразу на старте (нет весов, занят порт), без задержки крутил бы
    /// spawn в бесконечном цикле на полной скорости.
    private static let restartDelays: [TimeInterval] = [0, 1, 2, 5, 10, 30]
    private static let restartLimit = 10

    private var processes: [Int: Process] = [:]
    private var balancer: Process?
    private var shuttingDown = false
    private let queue = DispatchQueue(label: "oaitt.supervisor")

    /// Понятное сообщение вместо падения воркера: чего именно не хватает и где искали.
    static func missingWeights(settings: AppSettings) -> String? {
        let directory = URL(fileURLWithPath: settings.modelCacheDir)
            .appendingPathComponent(settings.modelType)
        let files = ["weights.safetensors", "tokenizer.model"]
        let absent = files.filter {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        guard !absent.isEmpty else { return nil }
        return "\(settings.modelType) weights not found in \(directory.path) - "
            + "download them in Settings > Models"
    }

    /// The CLI binary: next to the app inside a bundle, or in .build during development.
    static func executableURL(override: String) -> URL? {
        let fm = FileManager.default
        if !override.isEmpty, fm.isExecutableFile(atPath: override) {
            return URL(fileURLWithPath: override)
        }
        let candidates = [
            Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/oaitt-swift"),
            Bundle.main.bundleURL.deletingLastPathComponent()
                .appendingPathComponent("oaitt-swift"),
        ]
        return candidates.first { fm.isExecutableFile(atPath: $0.path) }
    }

    func start(settings: AppSettings) {
        guard state == .stopped || isFailed else { return }
        guard let executable = Self.executableURL(override: settings.cliPath) else {
            state = .failed("oaitt-swift binary not found; set its path in settings")
            return
        }

        // Проверяем веса до запуска: без этого воркер просто падал бы на старте, а
        // супервизор десять раз его поднимал и упирался в потолок с бесполезным
        // "crashed 11 times" вместо понятного "весов нет вот здесь".
        if let missing = Self.missingWeights(settings: settings) {
            state = .failed(missing)
            return
        }

        // Прошлый запуск мог оставить мёртвые процессы с висящими обработчиками -
        // иначе их terminationHandler снова уронил бы состояние в failed.
        shuttingDown = true
        queue.sync {
            for process in processes.values where process.isRunning { process.terminate() }
            processes.removeAll()
        }
        balancer?.terminate()
        balancer = nil

        shuttingDown = false
        state = .starting
        workers = []
        balancerPort = settings.port

        // Один воркер обслуживает порт сам: балансировщик перед ним только пересылал бы
        // байты (около 30-70 мс на 4 МБ), а на слабой машине воркер и будет один.
        let direct = settings.workerCount == 1
        let host = settings.bindAll ? "0.0.0.0" : "127.0.0.1"
        let ports =
            direct
            ? [settings.port]
            : (0..<settings.workerCount).map { settings.workerBasePort + $0 }

        for (index, port) in ports.enumerated() {
            var arguments = [
                "serve", "--host", direct ? host : "127.0.0.1", "--port", String(port),
                "--model-cache-dir", settings.modelCacheDir,
                "--model-type", settings.modelType,
            ]
            if settings.idleTimeout > 0 {
                arguments += ["--idle-timeout", String(settings.idleTimeout)]
            }
            if !settings.extraModels.isEmpty {
                arguments += ["--models", settings.extraModels]
            }
            arguments += ["--api-key", settings.apiKey]
            arguments += settings.telemetryEnabled ? ["--telemetry"] : ["--no-telemetry"]
            if settings.datasetEnabled {
                arguments += ["--dataset", "--dataset-limit-gb", String(settings.datasetLimitGb)]
            }
            if !settings.telemetryDir.isEmpty {
                arguments += ["--telemetry-dir", settings.telemetryDir]
            }
            arguments += ["--log-retention-days", String(settings.logRetentionDays)]
            let pid = launch(
                executable: executable, arguments: arguments, workerIndex: index, port: port)
            workers.append(
                WorkerInfo(
                    id: index, port: port, pid: pid, alive: true, restarts: 0, startedAt: Date()))
        }

        if !direct {
            var balancerArguments = [
                "balance", "--host", host, "--port", String(settings.port),
                "--backends", ports.map { "http://127.0.0.1:\($0)" }.joined(separator: ","),
            ]
            // Workers get the same token, so the balancer must also present it upstream.
            balancerArguments += ["--api-key", settings.apiKey, "--backend-key", settings.apiKey]
            balancer = spawn(executable: executable, arguments: balancerArguments)
        }
        state = .running
    }

    func stop() {
        shuttingDown = true
        queue.sync {
            for process in processes.values where process.isRunning { process.terminate() }
            processes.removeAll()
        }
        balancer?.terminate()
        balancer = nil
        workers = []
        state = .stopped
    }

    func restart(settings: AppSettings) {
        stop()
        start(settings: settings)
    }

    private var isFailed: Bool {
        if case .failed = state { return true }
        return false
    }

    @discardableResult
    private func launch(
        executable: URL, arguments: [String], workerIndex: Int, port: Int
    ) -> Int32 {
        let process = spawn(executable: executable, arguments: arguments)
        process.terminationHandler = { [weak self] _ in
            guard let self, !self.shuttingDown else { return }
            DispatchQueue.main.async {
                guard let slot = self.workers.firstIndex(where: { $0.id == workerIndex })
                else { return }
                self.workers[slot].alive = false
                self.workers[slot].restarts += 1

                let restarts = self.workers[slot].restarts
                guard restarts <= Self.restartLimit else {
                    self.state = .failed(
                        "worker on port \(port) crashed \(restarts) times, giving up")
                    return
                }

                // Живой воркер, проработавший минуту и упавший, поднимается сразу;
                // падающий на старте - с растущей задержкой.
                let healthy = Date().timeIntervalSince(self.workers[slot].startedAt) > 60
                let delay =
                    healthy
                    ? 0 : Self.restartDelays[min(restarts - 1, Self.restartDelays.count - 1)]
                if healthy { self.workers[slot].restarts = 1 }

                DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                    guard !self.shuttingDown,
                        let slot = self.workers.firstIndex(where: { $0.id == workerIndex })
                    else { return }
                    let pid = self.launch(
                        executable: executable, arguments: arguments, workerIndex: workerIndex,
                        port: port)
                    self.workers[slot].pid = pid
                    self.workers[slot].startedAt = Date()
                    self.workers[slot].alive = true
                }
            }
        }
        queue.sync { processes[workerIndex] = process }
        return process.processIdentifier
    }

    private func spawn(executable: URL, arguments: [String]) -> Process {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try? process.run()
        return process
    }
}
