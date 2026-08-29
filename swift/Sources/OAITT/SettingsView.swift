import SwiftUI

struct SettingsView: View {
    @Bindable var settings: AppSettings
    var supervisor: Supervisor

    var body: some View {
        TabView {
            serverTab.tabItem { Label("Server", systemImage: "server.rack") }
            modelTab.tabItem { Label("Models", systemImage: "waveform") }
            dataTab.tabItem { Label("Data", systemImage: "chart.bar") }
        }
        .frame(width: 520, height: 460)
        .padding(20)
    }

    private var serverTab: some View {
        Form {
            Section {
                Stepper(value: $settings.workerCount, in: 1...8) {
                    LabeledContent("Worker processes", value: "\(settings.workerCount)")
                }
                Text(
                    "MLX does not scale inside a single process - a pool of model copies "
                        + "there changes nothing. Throughput comes from more processes, each "
                        + "holding its own weights (about 850 MB)."
                )
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }

            Section("Network") {
                // LabeledContent с явной шириной: в .grouped длинный ярлык съедает
                // место под поле, и SecureField схлопывается в пустоту.
                LabeledContent("Port") {
                    TextField("", value: $settings.port, format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 90)
                }
                LabeledContent("First worker port") {
                    TextField("", value: $settings.workerBasePort, format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 90)
                }
                if !settings.portsAreValid {
                    Label(
                        "Balancer port falls inside the worker range",
                        systemImage: "exclamationmark.triangle"
                    )
                    .foregroundStyle(.orange).font(.caption)
                }
                Toggle("Accept connections from other machines", isOn: $settings.bindAll)
                LabeledContent("Bearer token") {
                    SecureField("empty - no auth", text: $settings.apiKey)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 220)
                }
            }

            Section {
                Toggle("Start serving when the app launches", isOn: $settings.startOnLaunch)
                TextField(
                    "Path to oaitt-swift (empty - look next to the app)", text: $settings.cliPath
                )
                .font(.caption)
                if supervisor.state == .running {
                    Button("Restart with new settings") { supervisor.restart(settings: settings) }
                    Text("Changes apply after a restart.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
    }

    private var modelTab: some View {
        Form {
            Section("Weights folder") {
                HStack {
                    Text(settings.modelCacheDir)
                        .font(.system(.caption, design: .monospaced))
                        .lineLimit(1).truncationMode(.head)
                        .textSelection(.enabled)
                    Spacer()
                    Button("Choose...") { pickModelDirectory() }
                    Button("Reveal") {
                        NSWorkspace.shared.selectFile(
                            nil, inFileViewerRootedAtPath: settings.modelCacheDir)
                    }
                }
            }

            Section("Models") {
                ModelListView(settings: settings)
            }

            Section("Serving") {
                Picker("Default model", selection: $settings.modelType) {
                    Text("CTC - faster").tag("ctc")
                    Text("RNNT - more accurate").tag("rnnt")
                }
                ServedModelsView(settings: settings)

                Stepper(value: $settings.idleTimeout, in: 0...3600, step: 60) {
                    LabeledContent(
                        "Unload weights after idle",
                        value: settings.idleTimeout == 0
                            ? "never" : "\(Int(settings.idleTimeout)) s")
                }
            }
        }
        .formStyle(.grouped)
    }

    private var dataTab: some View {
        Form {
            Section("Usage") {
                StatsView(settings: settings)
            }

            Section("Statistics") {
                Toggle("Record request statistics", isOn: $settings.telemetryEnabled)
                Text("Requests, transcribed seconds, memory. Local SQLite, kept 6 months.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section("Training set") {
                Toggle("Keep audio and transcripts", isOn: $settings.datasetEnabled)
                Text("Off by default. Everything stays on this machine.")
                    .font(.caption).foregroundStyle(.secondary)
                Stepper(value: $settings.datasetLimitGb, in: 1...500, step: 1) {
                    LabeledContent("Size limit", value: "\(Int(settings.datasetLimitGb)) GB")
                }
                Text("Oldest recordings are dropped when the limit is reached.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Request log") {
                Stepper(value: $settings.logRetentionDays, in: 0...30) {
                    LabeledContent(
                        "Keep logs",
                        value: settings.logRetentionDays == 0
                            ? "off" : "\(settings.logRetentionDays) days")
                }
                Text("One line per request: model, length, time, result. Open from the menu.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section("Storage") {
                HStack {
                    Text(
                        settings.telemetryDir.isEmpty
                            ? AppSettings.defaultTelemetryDir.path : settings.telemetryDir
                    )
                    .font(.system(.caption, design: .monospaced))
                    .lineLimit(1).truncationMode(.head)
                    Spacer()
                    Button("Reveal") {
                        let path =
                            settings.telemetryDir.isEmpty
                            ? AppSettings.defaultTelemetryDir.path : settings.telemetryDir
                        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    private func pickModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            settings.modelCacheDir = url.path
        }
    }
}
