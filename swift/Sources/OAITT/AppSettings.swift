import Foundation
import Observation
import SwiftUI

@Observable
final class AppSettings {
    /// More processes is the only way this scales: MLX serialises inside one.
    var workerCount: Int {
        didSet { store(workerCount, "workerCount") }
    }
    var port: Int { didSet { store(port, "port") } }
    var workerBasePort: Int { didSet { store(workerBasePort, "workerBasePort") } }
    var modelType: String { didSet { store(modelType, "modelType") } }
    var extraModels: String { didSet { store(extraModels, "extraModels") } }
    var modelCacheDir: String { didSet { store(modelCacheDir, "modelCacheDir") } }
    var apiKey: String { didSet { store(apiKey, "apiKey") } }
    var idleTimeout: Double { didSet { store(idleTimeout, "idleTimeout") } }
    var bindAll: Bool { didSet { store(bindAll, "bindAll") } }
    var startOnLaunch: Bool { didSet { store(startOnLaunch, "startOnLaunch") } }
    var cliPath: String { didSet { store(cliPath, "cliPath") } }
    var telemetryEnabled: Bool { didSet { store(telemetryEnabled, "telemetryEnabled") } }
    var datasetEnabled: Bool { didSet { store(datasetEnabled, "datasetEnabled") } }
    var datasetLimitGb: Double { didSet { store(datasetLimitGb, "datasetLimitGb") } }
    var telemetryDir: String { didSet { store(telemetryDir, "telemetryDir") } }
    /// Extra models the user added by HuggingFace repo, as JSON.
    var customModels: String { didSet { store(customModels, "customModels") } }
    var logRetentionDays: Int { didSet { store(logRetentionDays, "logRetentionDays") } }
    /// MLX keeps freed buffers around; on mixed chunk lengths that cache grows to gigabytes.
    var gpuCacheLimitMb: Int { didSet { store(gpuCacheLimitMb, "gpuCacheLimitMb") } }

    private let defaults = UserDefaults.standard

    init() {
        let d = UserDefaults.standard
        workerCount = d.object(forKey: "workerCount") as? Int ?? 1
        port = d.object(forKey: "port") as? Int ?? 9007
        workerBasePort = d.object(forKey: "workerBasePort") as? Int ?? 9010
        modelType = d.string(forKey: "modelType") ?? "rnnt"
        extraModels = d.string(forKey: "extraModels") ?? ""
        modelCacheDir = d.string(forKey: "modelCacheDir") ?? Self.defaultModelDir.path
        apiKey = d.string(forKey: "apiKey") ?? "key"
        idleTimeout = d.object(forKey: "idleTimeout") as? Double ?? 0
        bindAll = d.bool(forKey: "bindAll")
        startOnLaunch = d.object(forKey: "startOnLaunch") as? Bool ?? true
        cliPath = d.string(forKey: "cliPath") ?? ""
        telemetryEnabled = d.object(forKey: "telemetryEnabled") as? Bool ?? true
        datasetEnabled = d.bool(forKey: "datasetEnabled")
        datasetLimitGb = d.object(forKey: "datasetLimitGb") as? Double ?? 10
        telemetryDir = d.string(forKey: "telemetryDir") ?? ""
        customModels = d.string(forKey: "customModels") ?? "[]"
        logRetentionDays = d.object(forKey: "logRetentionDays") as? Int ?? 7
        gpuCacheLimitMb = d.object(forKey: "gpuCacheLimitMb") as? Int ?? 512
    }

    private func store(_ value: Any, _ key: String) {
        defaults.set(value, forKey: key)
    }

    static var defaultTelemetryDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("OAITT")
    }

    static var defaultModelDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("OAITT/models")
    }

    /// `extraModels` is a comma separated list on the wire (that is what the CLI takes),
    /// but a set in the UI - the checkboxes bind through here.
    func isServed(_ name: String) -> Bool {
        extraModels.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.contains(name)
    }

    func setServed(_ name: String, _ enabled: Bool) {
        var names = extraModels.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }
        names.removeAll { $0 == name }
        if enabled { names.append(name) }
        extraModels = names.joined(separator: ", ")
    }

    /// Worker ports must not collide with the balancer's.
    var portsAreValid: Bool {
        !(workerBasePort..<workerBasePort + workerCount).contains(port)
    }
}
