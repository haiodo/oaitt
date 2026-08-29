import GigaAM
import SwiftUI

/// Tail of today's request log.
///
/// The statistics answer "how much and how fast"; this answers "what came in and what came
/// back" - which is what you actually need when a transcript looks wrong.
@Observable
final class LogReader: @unchecked Sendable {
    private(set) var lines: [String] = []
    private(set) var file: URL?

    func reload(directory: String, limit: Int = 500) {
        let base =
            directory.isEmpty ? AppSettings.defaultTelemetryDir : URL(fileURLWithPath: directory)
        let logs = RequestLog.directory(for: base)

        let candidates =
            (try? FileManager.default.contentsOfDirectory(
                at: logs, includingPropertiesForKeys: [.contentModificationDateKey]))?
            .filter { $0.pathExtension == "log" }
            .sorted {
                let left =
                    (try? $0.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                let right =
                    (try? $1.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? .distantPast
                return left > right
            } ?? []

        guard let newest = candidates.first,
            let text = try? String(contentsOf: newest, encoding: .utf8)
        else {
            lines = []
            file = nil
            return
        }
        file = newest
        lines = text.split(separator: "\n").suffix(limit).map(String.init).reversed()
    }
}

struct LogView: View {
    var directory: String
    @State private var reader = LogReader()
    @State private var filter = ""

    private var shown: [String] {
        filter.isEmpty
            ? reader.lines
            : reader.lines.filter { $0.localizedCaseInsensitiveContains(filter) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                TextField("Filter", text: $filter)
                    .textFieldStyle(.roundedBorder)
                Button {
                    reader.reload(directory: directory)
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                if let file = reader.file {
                    Button("Reveal") {
                        NSWorkspace.shared.activateFileViewerSelecting([file])
                    }
                }
            }

            if shown.isEmpty {
                Text("No entries yet.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(shown.enumerated()), id: \.offset) { _, line in
                            Text(verbatim: line)
                                .font(.system(size: 11, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(6)
                }
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if let file = reader.file {
                Text(verbatim: file.lastPathComponent + " - newest first, kept 7 days")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(minWidth: 640, minHeight: 420)
        .onAppear { reader.reload(directory: directory) }
    }
}
