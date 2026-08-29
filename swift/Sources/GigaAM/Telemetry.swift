// Локальная телеметрия: статистика запросов и накопление датасета (аудио + тексты) на SQLite.

import Foundation
import SQLite3

// sqlite3_bind_text/blob по умолчанию не копируют переданный буфер - строка/массив Swift
// освобождается сразу после возврата из функции, и SQLite прочтёт уже мусор. TRANSIENT
// заставляет SQLite скопировать данные внутрь себя перед возвратом из bind-вызова.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public struct TelemetryConfig: Sendable {
    public var enabled: Bool
    public var datasetEnabled: Bool
    public var datasetLimitBytes: Int64
    public var retentionDays: Int
    public var directory: URL

    public init(
        enabled: Bool = true,
        datasetEnabled: Bool = false,
        datasetLimitBytes: Int64 = 10 * 1024 * 1024 * 1024,
        retentionDays: Int = 180,
        directory: URL
    ) {
        self.enabled = enabled
        self.datasetEnabled = datasetEnabled
        self.datasetLimitBytes = datasetLimitBytes
        self.retentionDays = retentionDays
        self.directory = directory
    }
}

public final class Telemetry: @unchecked Sendable {
    private enum BindValue {
        case int64(Int64)
        case double(Double)
        case text(String)
    }

    private let config: TelemetryConfig
    private let samplesDir: URL
    private var db: OpaquePointer?
    // Все обращения к db идут через lock - sqlite3_open_v2 открыт с FULLMUTEX (сериализация
    // и так есть на уровне самой SQLite), но lock нужен, чтобы последовательности из
    // нескольких запросов (например datasetBytes + DELETE в enforceLimit) были атомарны.
    private let lock = NSLock()
    private var lastCleanup = Date.distantPast

    public init?(config: TelemetryConfig) {
        guard config.enabled else { return nil }
        self.config = config
        self.samplesDir = config.directory.appendingPathComponent("samples", isDirectory: true)

        do {
            try FileManager.default.createDirectory(
                at: samplesDir, withIntermediateDirectories: true)
        } catch {
            Telemetry.logError("не удалось создать каталог \(samplesDir.path): \(error)")
            return nil
        }

        let dbPath = config.directory.appendingPathComponent("oaitt.sqlite").path
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(dbPath, &handle, flags, nil) == SQLITE_OK, let handle else {
            if let handle { sqlite3_close(handle) }
            Telemetry.logError("не удалось открыть базу \(dbPath)")
            return nil
        }
        db = handle

        // WAL: чтение (summary) не блокирует вставки от recordRequest/recordSample и наоборот.
        execLocked("PRAGMA journal_mode=WAL;")
        execLocked("PRAGMA synchronous=NORMAL;")
        execLocked(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                model TEXT NOT NULL,
                audio_seconds REAL NOT NULL,
                duration_ms REAL NOT NULL,
                status INTEGER NOT NULL,
                cached INTEGER NOT NULL,
                rss_mb REAL NOT NULL,
                gpu_mb REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                model TEXT NOT NULL,
                file TEXT NOT NULL,
                format TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
            """
        )

        cleanupExpiredLocked()
        lastCleanup = Date()
    }

    deinit {
        close()
    }

    public func close() {
        lock.withLock {
            guard let db else { return }
            sqlite3_close(db)
            self.db = nil
        }
    }

    public func recordRequest(
        model: String, audioSeconds: Double, durationMs: Double,
        status: Int, cached: Bool, rssMB: Double, gpuMB: Double
    ) {
        lock.withLock {
            execBindLocked(
                """
                INSERT INTO requests (ts, model, audio_seconds, duration_ms, status, cached, rss_mb, gpu_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    .int64(Int64(Date().timeIntervalSince1970)),
                    .text(model),
                    .double(audioSeconds),
                    .double(durationMs),
                    .int64(Int64(status)),
                    .int64(cached ? 1 : 0),
                    .double(rssMB),
                    .double(gpuMB),
                ]
            )
            maybeCleanupLocked()
        }
    }

    public func recordSample(model: String, audio: [UInt8], audioFormat: String, text: String) {
        lock.withLock {
            recordSampleLocked(model: model, audio: audio, audioFormat: audioFormat, text: text)
        }
    }

    public func datasetBytes() -> Int64 {
        lock.withLock { datasetBytesLocked() }
    }

    public func summary(sinceDays: Int) -> [String: Any] {
        lock.withLock { summaryLocked(sinceDays: sinceDays) }
    }

    /// Запросы по минутам за последние `minutes` минут, для графика в UI.
    /// Пустые минуты возвращаются нулями - иначе график врёт про паузы.
    public func requestsPerMinute(minutes: Int) -> [(minute: Date, count: Int)] {
        lock.withLock {
            let now = Int(Date().timeIntervalSince1970)
            let from = (now - minutes * 60) / 60 * 60
            var counts: [Int: Int] = [:]

            var stmt: OpaquePointer?
            defer { sqlite3_finalize(stmt) }
            let sql = """
                SELECT ts / 60 * 60 AS bucket, COUNT(*) FROM requests
                WHERE ts >= ? GROUP BY bucket;
                """
            if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
                sqlite3_bind_int64(stmt, 1, Int64(from))
                while sqlite3_step(stmt) == SQLITE_ROW {
                    counts[Int(sqlite3_column_int64(stmt, 0))] = Int(sqlite3_column_int64(stmt, 1))
                }
            }

            return stride(from: from, through: now / 60 * 60, by: 60).map { bucket in
                (Date(timeIntervalSince1970: TimeInterval(bucket)), counts[bucket] ?? 0)
            }
        }
    }

    // MARK: - recordSample

    private func recordSampleLocked(
        model: String, audio: [UInt8], audioFormat: String, text: String
    ) {
        guard config.datasetEnabled, db != nil else { return }
        let bytes = Int64(audio.count)

        // datasetBytesLocked - это SUM по всей таблице, поэтому считаем один раз.
        if datasetBytesLocked() + bytes > config.datasetLimitBytes {
            enforceLimitLocked(needed: bytes)
            // не влезло даже после чистки самых старых записей - пропускаем молча
            guard datasetBytesLocked() + bytes <= config.datasetLimitBytes else { return }
        }

        let filename = "\(UUID().uuidString).\(audioFormat)"
        let fileURL = samplesDir.appendingPathComponent(filename)
        do {
            try Data(audio).write(to: fileURL)
        } catch {
            Telemetry.logError("не удалось записать сэмпл \(fileURL.path): \(error)")
            return
        }

        let inserted = execBindLocked(
            "INSERT INTO samples (ts, model, file, format, bytes, text) VALUES (?, ?, ?, ?, ?, ?);",
            [
                .int64(Int64(Date().timeIntervalSince1970)),
                .text(model),
                .text(filename),
                .text(audioFormat),
                .int64(bytes),
                .text(text),
            ]
        )
        if !inserted {
            try? FileManager.default.removeItem(at: fileURL)
        }

        maybeCleanupLocked()
    }

    /// Удаляет самые старые сэмплы (запись + файл), пока needed байт не влезет в лимит.
    /// Удаляет самые старые сэмплы, пока не освободится место.
    ///
    /// Пачками, а не по одному: при лимите в гигабайты и мелких записях поштучное
    /// удаление давало бы тысячи SELECT под общим локом.
    private func enforceLimitLocked(needed: Int64) {
        guard let db else { return }
        var current = datasetBytesLocked()

        while current + needed > config.datasetLimitBytes {
            var victims: [(id: Int64, file: String, bytes: Int64)] = []
            var stmt: OpaquePointer?
            let sql = "SELECT id, file, bytes FROM samples ORDER BY ts ASC LIMIT 100;"
            if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
                while sqlite3_step(stmt) == SQLITE_ROW {
                    let file = sqlite3_column_text(stmt, 1).map { String(cString: $0) } ?? ""
                    victims.append(
                        (sqlite3_column_int64(stmt, 0), file, sqlite3_column_int64(stmt, 2)))
                }
            }
            sqlite3_finalize(stmt)
            guard !victims.isEmpty else { break }  // сэмплов больше нет, но всё равно не влезаем

            for victim in victims {
                execBindLocked("DELETE FROM samples WHERE id = ?;", [.int64(victim.id)])
                if !victim.file.isEmpty {
                    try? FileManager.default.removeItem(
                        at: samplesDir.appendingPathComponent(victim.file))
                }
                current -= victim.bytes
                if current + needed <= config.datasetLimitBytes { break }
            }
        }
    }

    private func datasetBytesLocked() -> Int64 {
        guard let db else { return 0 }
        let sql = "SELECT COALESCE(SUM(bytes), 0) FROM samples;"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            sqlite3_finalize(stmt)
            return 0
        }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return sqlite3_column_int64(stmt, 0)
    }

    // MARK: - TTL

    private func maybeCleanupLocked() {
        let now = Date()
        guard now.timeIntervalSince(lastCleanup) > 3600 else { return }
        cleanupExpiredLocked()
        lastCleanup = now
    }

    private func cleanupExpiredLocked() {
        guard let db else { return }
        let cutoff = Int64(Date().timeIntervalSince1970) - Int64(config.retentionDays) * 86400

        var filesToDelete: [String] = []
        let filesSql = "SELECT file FROM samples WHERE ts < ?;"
        var filesStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, filesSql, -1, &filesStmt, nil) == SQLITE_OK {
            sqlite3_bind_int64(filesStmt, 1, cutoff)
            while sqlite3_step(filesStmt) == SQLITE_ROW {
                if let cStr = sqlite3_column_text(filesStmt, 0) {
                    filesToDelete.append(String(cString: cStr))
                }
            }
        }
        sqlite3_finalize(filesStmt)

        execBindLocked("DELETE FROM samples WHERE ts < ?;", [.int64(cutoff)])
        execBindLocked("DELETE FROM requests WHERE ts < ?;", [.int64(cutoff)])

        for file in filesToDelete {
            try? FileManager.default.removeItem(at: samplesDir.appendingPathComponent(file))
        }
    }

    // MARK: - summary

    private func summaryLocked(sinceDays: Int) -> [String: Any] {
        var result: [String: Any] = [
            "requests": 0, "cached": 0, "errors": 0,
            "audio_seconds": 0.0, "avg_duration_ms": 0.0, "p95_duration_ms": 0.0,
            "realtime_factor": 0.0,
            "dataset_bytes": Int64(0), "dataset_samples": 0,
            "since_days": sinceDays,
        ]
        guard let db else { return result }
        let cutoff = Int64(Date().timeIntervalSince1970) - Int64(sinceDays) * 86400

        var requestsCount = 0
        let statsSql = """
            SELECT COUNT(*), COALESCE(SUM(cached), 0),
                   COALESCE(SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(audio_seconds), 0), COALESCE(AVG(duration_ms), 0),
                   COALESCE(SUM(duration_ms), 0)
            FROM requests WHERE ts >= ?;
            """
        var statsStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, statsSql, -1, &statsStmt, nil) == SQLITE_OK {
            sqlite3_bind_int64(statsStmt, 1, cutoff)
            if sqlite3_step(statsStmt) == SQLITE_ROW {
                requestsCount = Int(sqlite3_column_int64(statsStmt, 0))
                let sumDurationMs = sqlite3_column_double(statsStmt, 5)
                let audioSeconds = sqlite3_column_double(statsStmt, 3)
                result["requests"] = requestsCount
                result["cached"] = Int(sqlite3_column_int64(statsStmt, 1))
                result["errors"] = Int(sqlite3_column_int64(statsStmt, 2))
                result["audio_seconds"] = audioSeconds
                result["avg_duration_ms"] = sqlite3_column_double(statsStmt, 4)
                result["realtime_factor"] =
                    sumDurationMs > 0 ? audioSeconds / (sumDurationMs / 1000) : 0.0
            }
        } else {
            Telemetry.logError("summary: \(lastErrorMessage())")
        }
        sqlite3_finalize(statsStmt)

        if requestsCount > 0 {
            let offset = requestsCount * 95 / 100
            let p95Sql =
                "SELECT duration_ms FROM requests WHERE ts >= ? ORDER BY duration_ms LIMIT 1 OFFSET ?;"
            var p95Stmt: OpaquePointer?
            if sqlite3_prepare_v2(db, p95Sql, -1, &p95Stmt, nil) == SQLITE_OK {
                sqlite3_bind_int64(p95Stmt, 1, cutoff)
                sqlite3_bind_int64(p95Stmt, 2, Int64(offset))
                if sqlite3_step(p95Stmt) == SQLITE_ROW {
                    result["p95_duration_ms"] = sqlite3_column_double(p95Stmt, 0)
                }
            }
            sqlite3_finalize(p95Stmt)
        }

        result["dataset_bytes"] = datasetBytesLocked()

        var samplesStmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM samples;", -1, &samplesStmt, nil)
            == SQLITE_OK
        {
            if sqlite3_step(samplesStmt) == SQLITE_ROW {
                result["dataset_samples"] = Int(sqlite3_column_int64(samplesStmt, 0))
            }
        }
        sqlite3_finalize(samplesStmt)

        return result
    }

    // MARK: - SQLite helpers

    private func execLocked(_ sql: String) {
        guard let db else { return }
        if sqlite3_exec(db, sql, nil, nil, nil) != SQLITE_OK {
            Telemetry.logError("exec: \(lastErrorMessage())")
        }
    }

    /// Параметризованный INSERT/DELETE. Ошибка только логируется - вызывающий код
    /// (recordRequest/recordSample) не должен падать из-за проблем с базой.
    @discardableResult
    private func execBindLocked(_ sql: String, _ values: [BindValue]) -> Bool {
        guard let db else { return false }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            Telemetry.logError("prepare: \(lastErrorMessage())")
            sqlite3_finalize(stmt)
            return false
        }
        defer { sqlite3_finalize(stmt) }

        for (offset, value) in values.enumerated() {
            let idx = Int32(offset + 1)
            switch value {
            case .int64(let v): sqlite3_bind_int64(stmt, idx, v)
            case .double(let v): sqlite3_bind_double(stmt, idx, v)
            case .text(let v): sqlite3_bind_text(stmt, idx, v, -1, SQLITE_TRANSIENT)
            }
        }

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            Telemetry.logError("step: \(lastErrorMessage())")
            return false
        }
        return true
    }

    private func lastErrorMessage() -> String {
        guard let db else { return "no db" }
        return String(cString: sqlite3_errmsg(db))
    }

    private static func logError(_ message: String) {
        FileHandle.standardError.write(Data("[Telemetry] \(message)\n".utf8))
    }
}
