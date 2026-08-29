import SwiftUI

@main
struct OAITTApp: App {
    @State private var settings: AppSettings
    @State private var supervisor: Supervisor
    @State private var health: HealthPoller
    @State private var workers: WorkerMonitor

    /// MenuBarExtra builds its content lazily - only when the menu is opened - so
    /// autostart cannot live in a .task there, or nothing runs until the first click.
    init() {
        let settings = AppSettings()
        let supervisor = Supervisor()
        let health = HealthPoller()
        let workers = WorkerMonitor()
        _workers = State(initialValue: workers)
        _settings = State(initialValue: settings)
        _supervisor = State(initialValue: supervisor)
        _health = State(initialValue: health)

        if settings.startOnLaunch {
            supervisor.start(settings: settings)
            health.start(port: settings.port, apiKey: settings.apiKey)
            workers.start(supervisor: supervisor)
        }
    }

    var body: some Scene {
        MenuBarExtra("OAITT", systemImage: icon) {
            MenuContent(
                settings: settings, supervisor: supervisor, health: health, workers: workers)
        }
        .menuBarExtraStyle(.window)

        Settings {
            SettingsView(settings: settings, supervisor: supervisor)
        }
    }

    private var icon: String {
        switch supervisor.state {
        case .running: return health.inFlight > 0 ? "waveform.circle.fill" : "waveform.circle"
        case .starting: return "waveform.circle"
        case .failed: return "exclamationmark.triangle"
        case .stopped: return "waveform.slash"
        }
    }
}

struct MenuContent: View {
    @Bindable var settings: AppSettings
    var supervisor: Supervisor
    var health: HealthPoller
    var workers: WorkerMonitor
    @State private var showingLogs = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("OAITT").font(.headline)
                Spacer()
                Text(statusText).foregroundStyle(statusColor).font(.caption)
            }

            if supervisor.state == .running {
                // verbatim: interpolating an Int into a LocalizedStringKey groups it,
                // and the port showed up as "8 400".
                Text(verbatim: endpoint)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)

                Divider()

                WorkerListView(monitor: workers, inFlight: health.inFlight)
            }

            if supervisor.state == .running {
                Divider()
                TestRecorderView(settings: settings)
            }

            if supervisor.state == .running, settings.telemetryEnabled {
                Divider()
                MiniStatsView(directory: settings.telemetryDir)
            }

            if case .failed(let message) = supervisor.state {
                Text(message).font(.caption).foregroundStyle(.red).fixedSize(
                    horizontal: false, vertical: true)
            }

            Divider()

            HStack {
                Button(supervisor.state == .running ? "Stop" : "Start") {
                    if supervisor.state == .running {
                        supervisor.stop()
                        health.stop()
                        workers.stop()
                    } else {
                        supervisor.start(settings: settings)
                        health.start(port: settings.port, apiKey: settings.apiKey)
                        workers.start(supervisor: supervisor)
                    }
                }
                // SettingsLink is the only thing that reliably opens the Settings scene
                // from a MenuBarExtra; the old showSettingsWindow: selector is gone.
                Button("Logs") { showingLogs = true }
                SettingsLink { Text("Settings...") }
                    .simultaneousGesture(
                        TapGesture().onEnded {
                            NSApp.activate(ignoringOtherApps: true)
                        })
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }
            }
        }
        .padding(12)
        .frame(width: 300)
        .sheet(isPresented: $showingLogs) {
            VStack(alignment: .trailing, spacing: 0) {
                LogView(directory: settings.telemetryDir)
                Button("Close") { showingLogs = false }
                    .keyboardShortcut(.cancelAction)
                    .padding([.trailing, .bottom], 16)
            }
        }

    }

    private var endpoint: String {
        "http://\(settings.bindAll ? "0.0.0.0" : "127.0.0.1"):\(settings.port)/v1"
    }

    private var statusText: String {
        switch supervisor.state {
        case .running: return "running"
        case .starting: return "starting"
        case .stopped: return "stopped"
        case .failed: return "failed"
        }
    }

    private var statusColor: Color {
        switch supervisor.state {
        case .running: return .green
        case .starting: return .orange
        case .stopped: return .secondary
        case .failed: return .red
        }
    }
}
