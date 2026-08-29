// Several models in one process, chosen per request by the OpenAI `model` field.

import Foundation

public final class ModelRegistry: @unchecked Sendable {
    public struct Entry: Sendable {
        public let name: String
        public let modelType: GigaAMModelType
    }

    /// Names clients send in the `model` field.
    public static let known: [String: GigaAMModelType] = [
        "gigaam-ctc": .ctc,
        "gigaam-rnnt": .rnnt,
    ]

    public let names: [String]
    private let modelCacheDir: URL
    private let lockFree: Bool
    private let padBucketSec: Double
    private let idleTimeout: TimeInterval
    private let fallback: any ASREngine
    private var transcribers: [String: any ASREngine] = [:]
    private let lock = NSLock()

    public init(
        names: [String], fallback: any ASREngine, modelCacheDir: URL,
        lockFree: Bool = true, padBucketSec: Double = 1.0, idleTimeout: TimeInterval = 0
    ) {
        self.names = names.filter { Self.known[$0] != nil }
        self.fallback = fallback
        self.modelCacheDir = modelCacheDir
        self.lockFree = lockFree
        self.padBucketSec = padBucketSec
        self.idleTimeout = idleTimeout
    }

    /// Loaded on first use, not at startup - keep in memory only what is asked for.
    /// An unknown name falls back to the default model: clients send both `whisper-1`
    /// and `gigaam`, and neither should fail the request.
    public func transcriber(for name: String?) -> any ASREngine {
        let key = (name ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        guard names.contains(key), let type = Self.known[key] else { return fallback }

        return lock.withLock {
            if let existing = transcribers[key] { return existing }
            guard
                let loaded = try? GigaAMTranscriber(
                    modelDir: modelCacheDir.appendingPathComponent(type.rawValue),
                    modelType: type, lockFree: lockFree, padBucketSec: padBucketSec,
                    idleTimeout: idleTimeout)
            else { return fallback }
            transcribers[key] = loaded
            return loaded
        }
    }

    public var loaded: [String] {
        lock.withLock { transcribers.keys.sorted() }
    }
}
