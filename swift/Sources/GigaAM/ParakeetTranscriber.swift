// Parakeet TDT as an ASREngine: loads an mlx-community/parakeet-tdt-0.6b-v3 directory
// (config.json + model.safetensors), chunks audio like GigaAMTranscriber and emits
// Segments with word timestamps.

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
    private let castToBF16: Bool
    private var model: ParakeetTDTModel?
    private let loadLock = NSLock()
    private var lastActivity = Date()

    public var isLoaded: Bool { loadLock.withLock { model != nil } }

    public init(
        modelDir: URL, idleTimeout: TimeInterval = 0,
        maxChunkSec: Double = 20.0, castToBF16: Bool = false
    ) throws {
        self.modelDir = modelDir
        self.castToBF16 = castToBF16
        self.idleTimeout = idleTimeout
        self.maxChunkSec = maxChunkSec

        let fm = FileManager.default
        guard fm.fileExists(atPath: modelDir.appendingPathComponent("model.safetensors").path),
            fm.fileExists(atPath: modelDir.appendingPathComponent("config.json").path)
        else {
            throw AudioError.decodeFailed(
                "model.safetensors/config.json missing in \(modelDir.path); "
                    + "expected the mlx-community/parakeet-tdt-0.6b-v3 layout")
        }
        _ = try loaded()
        startIdleMonitor()
    }

    private func loaded() throws -> ParakeetTDTModel {
        try loadLock.withLock {
            if let model { return model }

            let config = try JSONSerialization.jsonObject(
                with: Data(contentsOf: modelDir.appendingPathComponent("config.json")))
            guard let root = config as? [String: Any],
                let pre = root["preprocessor"] as? [String: Any],
                let joint = root["joint"] as? [String: Any],
                let vocab = joint["vocabulary"] as? [String],
                let sampleRate = pre["sample_rate"] as? Int,
                let nFFT = pre["n_fft"] as? Int,
                let features = pre["features"] as? Int
            else {
                throw AudioError.decodeFailed("config.json lacks preprocessor/joint.vocabulary")
            }
            // The mel and STFT code uses fixed constants; refuse a config they do not match.
            guard sampleRate == ParakeetMel.sampleRate, nFFT == ParakeetMel.nFFT,
                features == ParakeetMel.nMels
            else {
                throw AudioError.decodeFailed("unsupported preprocessor settings in config.json")
            }
            let filterbanks = ParakeetMel.slaneyFilterbank(
                sampleRate: sampleRate, nFFT: nFFT, nMels: features)
            let network = ParakeetTDTModel(vocabulary: vocab, filterbanks: filterbanks)
            var weights = try MLX.loadArrays(
                url: modelDir.appendingPathComponent("model.safetensors"))
            if castToBF16 {
                weights = weights.mapValues { $0.asType(.bfloat16) }
            }
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
