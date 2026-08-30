import SwiftUI

struct WorkerListView: View {
    var monitor: WorkerMonitor
    var inFlight: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Workers").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if inFlight > 0 {
                    Text(verbatim: "\(inFlight) in flight")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            ForEach(monitor.rows) { row in
                HStack(spacing: 6) {
                    Circle()
                        .fill(color(for: row))
                        .frame(width: 6, height: 6)

                    Text(verbatim: ":\(row.port)")
                        .font(.system(.caption2, design: .monospaced))
                        .frame(width: 42, alignment: .leading)

                    Text(verbatim: String(format: "%.0f%%", row.cpuPercent))
                        .font(.caption2).monospacedDigit()
                        .frame(width: 36, alignment: .trailing)

                    Text(verbatim: memoryText(row))
                        .font(.caption2).foregroundStyle(.secondary)
                        .lineLimit(1).fixedSize()

                    Spacer()

                    Text(verbatim: uptimeText(row))
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .help(tooltip(for: row))
            }
        }
    }

    /// Green only when the worker answers and holds its weights: a process that is up but
    /// has unloaded the model after idle is a different state than one serving requests.
    private func color(for row: WorkerMonitor.Row) -> Color {
        if !row.responding { return .red }
        return row.modelLoaded ? .green : .orange
    }

    /// Показываем footprint, как Activity Monitor, и отдельно то, что MLX держит на GPU -
    /// второе входит в первое, но его полезно видеть само по себе.
    private func memoryText(_ row: WorkerMonitor.Row) -> String {
        // Обе величины в гигабайтах: в мегабайтах строка не влезала в ширину попапа и
        // обрывалась на "865 MB · 848...".
        func short(_ mb: Double) -> String {
            mb >= 1024 ? String(format: "%.1fG", mb / 1024) : String(format: "%.0fM", mb)
        }
        return row.gpuMB > 0
            ? "\(short(row.footprintMB)) · \(short(row.gpuMB)) gpu" : short(row.footprintMB)
    }

    private func uptimeText(_ row: WorkerMonitor.Row) -> String {
        let seconds = Int(row.uptime)
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 { return "\(seconds / 60)m" }
        return "\(seconds / 3600)h \((seconds % 3600) / 60)m"
    }

    private func tooltip(for row: WorkerMonitor.Row) -> String {
        var parts = ["pid \(row.pid)"]
        parts.append(row.responding ? "responding" : "not responding")
        parts.append(row.modelLoaded ? "weights loaded" : "weights unloaded")
        if row.idleTimeout > 0 {
            parts.append("unloads after \(Int(row.idleTimeout))s idle")
        }
        if row.restarts > 0 {
            parts.append("restarted \(row.restarts)x")
        }
        return parts.joined(separator: ", ")
    }
}
