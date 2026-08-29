import Charts
import GigaAM
import SwiftUI

/// Reads the same SQLite the workers write to. They all share one directory, so one
/// summary covers the whole pool.
@Observable
final class StatsReader: @unchecked Sendable {
    private(set) var summary: [String: Any] = [:]
    private(set) var rate: [(minute: Date, count: Int)] = []
    private(set) var available = false

    func reloadRate(directory: String, minutes: Int = 30) {
        let url =
            directory.isEmpty ? AppSettings.defaultTelemetryDir : URL(fileURLWithPath: directory)
        guard let telemetry = Telemetry(config: TelemetryConfig(directory: url)) else { return }
        rate = telemetry.requestsPerMinute(minutes: minutes)
        summary = telemetry.summary(sinceDays: 1)
        available = true
        telemetry.close()
    }

    func reload(directory: String, days: Int) {
        let url =
            directory.isEmpty ? AppSettings.defaultTelemetryDir : URL(fileURLWithPath: directory)
        guard let telemetry = Telemetry(config: TelemetryConfig(directory: url)) else {
            available = false
            return
        }
        summary = telemetry.summary(sinceDays: days)
        available = true
        telemetry.close()
    }

    func int(_ key: String) -> Int {
        (summary[key] as? Int) ?? Int((summary[key] as? Int64) ?? 0)
    }

    func double(_ key: String) -> Double {
        (summary[key] as? Double) ?? Double(int(key))
    }
}

struct StatsView: View {
    @Bindable var settings: AppSettings
    @State private var reader = StatsReader()
    @State private var days = 30

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Picker("Period", selection: $days) {
                    Text("24 hours").tag(1)
                    Text("7 days").tag(7)
                    Text("30 days").tag(30)
                    Text("6 months").tag(180)
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Button {
                    reader.reload(directory: settings.telemetryDir, days: days)
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }

            if reader.available {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    row("Requests", "\(reader.int("requests"))")
                    row("From cache", "\(reader.int("cached"))")
                    row("Errors", "\(reader.int("errors"))")
                    row("Audio transcribed", formatDuration(reader.double("audio_seconds")))
                    row("Speed", String(format: "%.0fx realtime", reader.double("realtime_factor")))
                    row("Average", String(format: "%.0f ms", reader.double("avg_duration_ms")))
                    row("p95", String(format: "%.0f ms", reader.double("p95_duration_ms")))
                    if reader.int("dataset_samples") > 0 {
                        row(
                            "Kept recordings",
                            "\(reader.int("dataset_samples")), "
                                + formatBytes(reader.int("dataset_bytes")))
                    }
                }
            } else {
                Text("No statistics yet.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .onAppear { reader.reload(directory: settings.telemetryDir, days: days) }
        .onChange(of: days) { reader.reload(directory: settings.telemetryDir, days: days) }
    }

    @ViewBuilder
    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(verbatim: value).font(.caption).monospacedDigit()
        }
    }

    private func formatDuration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0f s", seconds) }
        if seconds < 3600 { return String(format: "%.0f min", seconds / 60) }
        return String(format: "%.1f h", seconds / 3600)
    }

    private func formatBytes(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1024 / 1024
        return mb < 1024 ? String(format: "%.0f MB", mb) : String(format: "%.1f GB", mb / 1024)
    }
}

/// Compact panel for the menu bar: today's numbers plus requests per minute.
struct MiniStatsView: View {
    var directory: String
    @State private var reader = StatsReader()

    private let refresh = Timer.publish(every: 10, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if reader.available, reader.int("requests") > 0 {
                HStack(spacing: 12) {
                    stat("\(reader.int("requests"))", "today")
                    stat(String(format: "%.0fx", reader.double("realtime_factor")), "realtime")
                    stat(String(format: "%.0f ms", reader.double("p95_duration_ms")), "p95")
                }

                Chart(reader.rate, id: \.minute) { point in
                    BarMark(
                        x: .value("Minute", point.minute, unit: .minute),
                        y: .value("Requests", point.count)
                    )
                    .foregroundStyle(.tint)
                }
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 2))
                }
                .frame(height: 46)

                Text("requests per minute, last 30 min")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                Text("No requests yet.").font(.caption).foregroundStyle(.secondary)
            }
        }
        .onAppear { reader.reloadRate(directory: directory) }
        .onReceive(refresh) { _ in reader.reloadRate(directory: directory) }
    }

    @ViewBuilder
    private func stat(_ value: String, _ label: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(verbatim: value).font(.caption).monospacedDigit().bold()
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }
}
