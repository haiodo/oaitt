// Один файл на день в <directory>/logs, старые удаляются.
//
// Телеметрия отвечает на вопрос "сколько и как быстро", лог - на вопрос "что именно
// пришло и что вернулось". Второе нужно, когда расшифровка выглядит неправильно и надо
// понять, на каком запросе это случилось.

import Foundation

public final class RequestLog: @unchecked Sendable {
    private let directory: URL
    private let retentionDays: Int
    private let lock = NSLock()
    private var handle: FileHandle?
    private var openedDay: String = ""

    private static let dayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.timeZone = .current
        return formatter
    }()

    private static let stampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        formatter.timeZone = .current
        return formatter
    }()

    public init?(directory: URL, retentionDays: Int = 7) {
        self.directory = directory.appendingPathComponent("logs")
        self.retentionDays = retentionDays
        guard
            (try? FileManager.default.createDirectory(
                at: self.directory, withIntermediateDirectories: true)) != nil
        else { return nil }
        removeExpired()
    }

    public static func directory(for base: URL) -> URL {
        base.appendingPathComponent("logs")
    }

    public func record(
        model: String, audioSeconds: Double, durationMs: Double, status: Int, cached: Bool,
        text: String
    ) {
        let now = Date()
        let excerpt = text.replacingOccurrences(of: "\n", with: " ").prefix(120)
        let line = String(
            format: "%@ %-5@ %6.1fs %6.0fms %d%@ %@\n",
            Self.stampFormatter.string(from: now), model, audioSeconds, durationMs, status,
            cached ? " cached" : "", String(excerpt))

        lock.withLock {
            let day = Self.dayFormatter.string(from: now)
            // Переоткрываем и когда сменились сутки, и когда файл унесли из-под нас:
            // иначе записи молча уходили бы в удалённый инод до следующей полуночи.
            let path = directory.appendingPathComponent("oaitt-\(day).log").path
            if day != openedDay || !FileManager.default.fileExists(atPath: path) {
                reopen(day: day)
            }
            guard let handle, let data = line.data(using: .utf8) else { return }
            try? handle.write(contentsOf: data)
        }
    }

    public func close() {
        lock.withLock {
            try? handle?.close()
            handle = nil
        }
    }

    /// Смена суток закрывает вчерашний файл и заодно чистит просроченные.
    private func reopen(day: String) {
        try? handle?.close()
        let url = directory.appendingPathComponent("oaitt-\(day).log")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: url)
        try? handle?.seekToEnd()
        openedDay = day
        removeExpired()
    }

    private func removeExpired() {
        let cutoff = Date().addingTimeInterval(-Double(retentionDays) * 86400)
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for file in files where file.pathExtension == "log" {
            let modified = (try? file.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate
            if let modified, modified < cutoff {
                try? FileManager.default.removeItem(at: file)
            }
        }
    }
}
