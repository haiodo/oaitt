// Parakeet TDT as an ASREngine: loads data/parakeet_tdt_v3 (weights.safetensors,
// filterbanks.safetensors, vocab.txt from scripts/convert_parakeet_to_mlx_swift.py),
// chunks audio like GigaAMTranscriber and emits Segments with word timestamps.

import Foundation
import MLX
import MLXNN

public struct ParakeetWord: Codable, Sendable, Equatable {
    public let word: String
    public let start: Double
    public let end: Double
    public let probability: Float
}

public final class ParakeetTranscriber: ASREngine, @unchecked Sendable {
    public var name: String { "parakeet-tdt-v3" }
    /// Single inference stream per process, same as the Python engine on mlx 0.32.x.
    public let lockFree = true
    public var padBucketSec: Double { 0 }
    public let idleTimeout: TimeInterval
    let maxChunkSec: Double

    private let modelDir: URL
    private var model: ParakeetTDTModel?
    private let loadLock = NSLock()
    private var lastActivity = Date()

    public var isLoaded: Bool { loadLock.withLock { model != nil } }

    public init(
        modelDir: URL, idleTimeout: TimeInterval = 0,
        maxChunkSec: Double = 20.0
    ) throws {
        self.modelDir = modelDir
        self.idleTimeout = idleTimeout
        self.maxChunkSec = maxChunkSec

        let fm = FileManager.default
        let weights = modelDir.appendingPathComponent("weights.safetensors")
        let banks = modelDir.appendingPathComponent("filterbanks.safetensors")
        let vocab = modelDir.appendingPathComponent("vocab.txt")
        guard fm.fileExists(atPath: weights.path),
            fm.fileExists(atPath: banks.path),
            fm.fileExists(atPath: vocab.path)
        else {
            throw AudioError.decodeFailed(
                "parakeet weights/filterbanks/vocab missing in \(modelDir.path); "
                    + "run scripts/convert_parakeet_to_mlx_swift.py")
        }
        _ = try loaded()
        startIdleMonitor()
    }

    private func loaded() throws -> ParakeetTDTModel {
        try loadLock.withLock {
            if let model { return model }

            // The file ends with a newline; the trailing empty line is not a token.
            let vocab = try String(
                contentsOf: modelDir.appendingPathComponent("vocab.txt"), encoding: .utf8
            )
            .components(separatedBy: "\n").filter { !$0.isEmpty }
            let banks = try MLX.loadArrays(
                url: modelDir.appendingPathComponent("filterbanks.safetensors"))
            guard let filterbanks = banks["filterbanks"] else {
                throw AudioError.decodeFailed("filterbanks.safetensors has no 'filterbanks' key")
            }
            let network = ParakeetTDTModel(vocabulary: vocab, filterbanks: filterbanks)
            let weights = try MLX.loadArrays(
                url: modelDir.appendingPathComponent("weights.safetensors"))
            try network.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: .all)
            network.train(false)
            eval(network)

            model = network
            return network
        }
    }

    public func release() {
        loadLock.withLock {
            model = nil
        }
        MLX.GPU.clearCache()
    }

    private func startIdleMonitor() {
        guard idleTimeout > 0 else { return }
        Thread.detachNewThread { [weak self] in
            while let self {
                Thread.sleep(forTimeInterval: min(15, self.idleTimeout))
                let idle = Date().timeIntervalSince(self.loadLock.withLock { self.lastActivity })
                if idle > self.idleTimeout, self.isLoaded {
                    self.release()
                }
            }
        }
    }

    public func transcribe(audio: [Float], maxChunkSec chunkOverride: Double) -> [Segment] {
        let resolvedChunkSec = chunkOverride > 0 ? chunkOverride : self.maxChunkSec
        let sr = Double(ParakeetMel.sampleRate)

        loadLock.withLock { lastActivity = Date() }
        guard let model = try? loaded() else { return [] }

        var segments: [Segment] = []
        for chunk in splitAudio(audio, maxChunkSec: resolvedChunkSec) {
            // No bucket padding, unlike GigaAM: the per-feature mel normalization and
            // unmasked attention see the padding, and Golos WER went from 3.99% to 4.41%.
            let slice = Array(audio[chunk.startSample..<chunk.endSample])
            guard slice.count >= ParakeetMel.winLength else { continue }

            let tokens = model.transcribeChunk(slice)
            let offset = Double(chunk.startSample) / sr
            let realEnd = Double(slice.count) / sr
            segments.append(
                contentsOf: Self.segments(from: tokens, offset: offset, realEnd: realEnd))
        }
        return segments
    }

    /// Groups tokens into sentences (punctuation rule from the Python port) and words
    /// (a piece starting with a space opens a word; probability = min over pieces).
    /// Token times are chunk-relative; `offset` places them on the file timeline and
    /// `realEnd` keeps the last segment inside the chunk.
    static func segments(
        from tokens: [ParakeetToken], offset: Double, realEnd: Double
    ) -> [Segment] {
        var sentences: [[ParakeetToken]] = []
        var current: [ParakeetToken] = []
        for (idx, token) in tokens.enumerated() {
            current.append(token)
            let isPunctuation =
                token.text.contains("!")
                || token.text.contains("?")
                || token.text.contains("。")
                || token.text.contains("？")
                || token.text.contains("！")
                || (token.text.contains(".")
                    && (idx == tokens.count - 1 || tokens[idx + 1].text.contains(" ")))
            if isPunctuation {
                sentences.append(current)
                current = []
            }
        }
        if !current.isEmpty {
            sentences.append(current)
        }

        return sentences.compactMap { sentence in
            let text = sentence.map(\.text).joined().trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { return nil }

            let start = offset + sentence[0].start
            let end = min(offset + sentence[sentence.count - 1].end, offset + realEnd)

            var words: [ParakeetWord] = []
            var word: PendingWord?
            for token in sentence {
                if token.text.hasPrefix(" ") || word == nil {
                    if let w = word {
                        words.append(w.parakeetWord)
                    }
                    word = PendingWord(
                        text: token.text.trimmingCharacters(in: .whitespaces),
                        start: offset + token.start,
                        end: offset + token.end,
                        probability: token.confidence)
                } else if var w = word {
                    w.text += token.text
                    w.end = offset + token.end
                    w.probability = min(w.probability, token.confidence)
                    word = w
                }
            }
            if let w = word {
                words.append(w.parakeetWord)
            }

            return Segment(
                start: start, end: end, text: text,
                words: words.isEmpty ? nil : words)
        }
    }
}

/// Word under construction while grouping tokenizer pieces; a struct keeps
/// swiftlint's large-tuple rule happy and the mutation explicit.
private struct PendingWord {
    var text: String
    var start: Double
    var end: Double
    var probability: Float

    var parakeetWord: ParakeetWord {
        ParakeetWord(word: text, start: start, end: end, probability: probability)
    }
}
