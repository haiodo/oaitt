// Cache of transcripts keyed by the audio itself.
//
// The platform retries tasks - up to 5 times on transient errors and without a limit on
// network ones - and every retry brings the same chunk back. Keying on a hash of the audio
// plus the request parameters means a hit is literally the same request, not a guess.

import CryptoKit
import Foundation

public final class ResultCache: @unchecked Sendable {
    private struct Entry {
        let storedAt: Date
        let value: (segments: [Segment], audioSeconds: Double)
    }

    private let maxEntries: Int
    private let ttl: TimeInterval
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    private let lock = NSLock()
    private(set) var hits = 0
    private(set) var misses = 0

    public var isEnabled: Bool { maxEntries > 0 && ttl > 0 }

    public init(maxEntries: Int = 256, ttl: TimeInterval = 300) {
        self.maxEntries = maxEntries
        self.ttl = ttl
    }

    /// Hashes the encoded bytes rather than decoded samples: a retry re-sends the very
    /// same file, and this way a hit costs nothing - no temp file, no decode.
    public static func key(bytes: [UInt8], parameters: [String: String]) -> String {
        var hasher = SHA256()
        bytes.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        for name in parameters.keys.sorted() {
            hasher.update(data: Data("|\(name)=\(parameters[name] ?? "")".utf8))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public func get(_ key: String) -> (segments: [Segment], audioSeconds: Double)? {
        guard isEnabled else { return nil }
        return lock.withLock {
            guard let entry = entries[key] else {
                misses += 1
                return nil
            }
            guard Date().timeIntervalSince(entry.storedAt) <= ttl else {
                remove(key)
                misses += 1
                return nil
            }
            touch(key)
            hits += 1
            return entry.value
        }
    }

    public func put(_ key: String, _ value: (segments: [Segment], audioSeconds: Double)) {
        guard isEnabled else { return }
        lock.withLock {
            entries[key] = Entry(storedAt: Date(), value: value)
            touch(key)
            while order.count > maxEntries {
                entries[order.removeFirst()] = nil
            }
        }
    }

    /// Свежее обращение переносит ключ в конец очереди вытеснения - без этого кеш вёл
    /// себя как FIFO и выбрасывал как раз то, что чаще всего спрашивают.
    private func touch(_ key: String) {
        if let index = order.firstIndex(of: key) { order.remove(at: index) }
        order.append(key)
    }

    private func remove(_ key: String) {
        entries[key] = nil
        if let index = order.firstIndex(of: key) { order.remove(at: index) }
    }

    public var stats: [String: Any] {
        lock.withLock {
            let total = hits + misses
            var out: [String: Any] = [
                "enabled": isEnabled,
                "entries": entries.count,
                "max_entries": maxEntries,
                "ttl_sec": ttl,
                "hits": hits,
                "misses": misses,
            ]
            if total > 0 {
                out["hit_rate"] = (Double(hits) / Double(total) * 10000).rounded() / 10000
            }
            return out
        }
    }
}
