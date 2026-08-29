// Common surface for speech recognition backends.
//
// The server, the registry and the CLI only ever need "audio in, segments out" plus a way
// to release the weights. Keeping that in a protocol is what lets a second architecture -
// multilingual GigaAM, Whisper, Parakeet - sit next to GigaAM v3 without touching them.

import Foundation

public protocol ASREngine: AnyObject, Sendable {
    /// Model variant as it is reported in /health and /v1/models.
    var name: String { get }

    /// False after an idle unload; the next transcribe call reloads.
    var isLoaded: Bool { get }

    /// Seconds of inactivity after which weights are dropped; 0 keeps them loaded.
    var idleTimeout: TimeInterval { get }

    func transcribe(audio: [Float], maxChunkSec: Double) -> [Segment]

    /// Drops the weights and the MLX buffer cache.
    func release()
}

extension ASREngine {
    public func transcribe(audio: [Float]) -> [Segment] {
        transcribe(audio: audio, maxChunkSec: 20.0)
    }

    public func transcribe(url: URL, maxChunkSec: Double = 20.0) throws -> [Segment] {
        transcribe(audio: try AudioLoader.load(url: url), maxChunkSec: maxChunkSec)
    }
}
