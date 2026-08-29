import Foundation
import MLX
import MLXNN

public struct Segment: Codable, Sendable, Equatable {
    public let start: Double
    public let end: Double
    public let text: String
}

public final class GigaAMTranscriber: ASREngine, @unchecked Sendable {
    public let modelType: GigaAMModelType
    public var name: String { modelType.rawValue }
    private let modelDir: URL
    private let weightsURL: URL
    private let tokenizerURL: URL
    private var model: GigaAMMLX?
    private var tokenizer: SentencePieceTokenizer?
    private let mel = MelSpectrogram()
    private let lock = NSLock()
    private let loadLock = NSLock()
    private var lastActivity = Date()
    /// Seconds of inactivity after which the weights are dropped; 0 keeps them forever.
    /// A GigaAM model holds about 850 MB of unified memory, which is worth reclaiming on a
    /// laptop that is not transcribing anything right now.
    public let idleTimeout: TimeInterval
    /// MLX ops are lazy and thread-safe to build; the GPU runtime serialises execution,
    /// so parallel clients scale without extra model copies. Turn off when unified
    /// memory is tight - each in-flight request holds its own activations.
    public let lockFree: Bool
    /// Chunk lengths are rounded up to whole `padBucketSec` steps before the encoder runs.
    /// MLX compiles kernels and caches buffers per tensor shape, so the raw VAD split -
    /// a new length for every chunk - defeats both. 0 disables padding.
    public let padBucketSec: Double

    public var isLoaded: Bool { loadLock.withLock { model != nil } }

    public init(
        modelDir: URL, modelType: GigaAMModelType, lockFree: Bool = true,
        padBucketSec: Double = 1.0, idleTimeout: TimeInterval = 0
    ) throws {
        self.modelType = modelType
        self.modelDir = modelDir
        self.lockFree = lockFree
        self.padBucketSec = padBucketSec
        self.idleTimeout = idleTimeout

        let fm = FileManager.default
        let weights = [
            modelDir.appendingPathComponent("weights_\(modelType.rawValue).safetensors"),
            modelDir.appendingPathComponent("weights.safetensors"),
        ].first { fm.fileExists(atPath: $0.path) }
        let tokenizerPath = [
            modelDir.appendingPathComponent("tokenizer_\(modelType.rawValue).model"),
            modelDir.appendingPathComponent("tokenizer.model"),
        ].first { fm.fileExists(atPath: $0.path) }

        guard let weights, let tokenizerPath else {
            throw AudioError.decodeFailed("weights or tokenizer missing in \(modelDir.path)")
        }

        self.weightsURL = weights
        self.tokenizerURL = tokenizerPath
        _ = try loaded()
        startIdleMonitor()
    }

    /// Loads the weights on first use and after an idle unload.
    private func loaded() throws -> (GigaAMMLX, SentencePieceTokenizer) {
        try loadLock.withLock {
            if let model, let tokenizer { return (model, tokenizer) }

            let network = GigaAMMLX(modelType: modelType)
            try network.update(
                parameters: ModuleParameters.unflattened(MLX.loadArrays(url: weightsURL)),
                verify: .all)
            eval(network)
            let pieces = try SentencePieceTokenizer(path: tokenizerURL)
            model = network
            tokenizer = pieces
            return (network, pieces)
        }
    }

    /// Drops the weights and the MLX buffer cache.
    public func release() {
        loadLock.withLock {
            model = nil
            tokenizer = nil
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

    public func transcribe(audio: [Float], maxChunkSec: Double) -> [Segment] {
        let sr = Double(MelSpectrogram.sampleRate)
        let maxSamples = Int(maxChunkSec * sr)
        let bucketSamples = padBucketSec > 0 ? Int(padBucketSec * sr) : 0

        var slices: [[Float]] = []
        var realLengths: [Int] = []
        var bounds: [(start: Double, end: Double)] = []

        for chunk in splitAudio(audio, maxChunkSec: maxChunkSec) {
            var slice = Array(audio[chunk.startSample..<chunk.endSample])
            guard slice.count >= MelSpectrogram.nFFT else { continue }

            realLengths.append(slice.count)
            if bucketSamples > 0 {
                let target = min(
                    maxSamples, (slice.count + bucketSamples - 1) / bucketSamples * bucketSamples)
                if slice.count < target {
                    slice.append(contentsOf: repeatElement(0, count: target - slice.count))
                }
            }
            slices.append(slice)
            bounds.append(
                (Double(chunk.startSample) / sr, Double(chunk.endSample) / sr))
        }
        guard !slices.isEmpty else { return [] }

        loadLock.withLock { lastActivity = Date() }
        guard let (model, tokenizer) = try? loaded() else { return [] }
        let texts =
            lockFree
            ? run(slices, realLengths, model, tokenizer)
            : lock.withLock { run(slices, realLengths, model, tokenizer) }

        return zip(texts, bounds).compactMap { text, bound in
            text.isEmpty ? nil : Segment(start: bound.start, end: bound.end, text: text)
        }
    }

    /// Encodes every chunk, then decodes them together - RNNT greedy costs one GPU sync per
    /// frame, so decoding the chunks as a batch amortises those syncs across the whole request.
    private func run(
        _ slices: [[Float]], _ realLengths: [Int], _ model: GigaAMMLX,
        _ tokenizer: SentencePieceTokenizer
    ) -> [String] {
        // CTC decodes with a single argmax, so encoding and decoding chunk by chunk keeps the
        // GPU pipelined. RNNT pays one GPU sync per frame, so its chunks are decoded as a batch.
        if modelType == .ctc {
            return zip(slices, realLengths).map { slice, realLength in
                let (encoded, paddedSeqLen) = model.encode(mel(slice).expandedDimensions(axis: 0))
                eval(encoded)
                let seqLen = min(paddedSeqLen, encoderFrames(realLength))
                return tokenizer.decode(model.decode(encoded, seqLen: seqLen))
            }
        }

        var encodedList: [MLXArray] = []
        var seqLens: [Int] = []
        for (slice, realLength) in zip(slices, realLengths) {
            let (encoded, paddedSeqLen) = model.encode(mel(slice).expandedDimensions(axis: 0))
            encodedList.append(encoded)
            seqLens.append(min(paddedSeqLen, encoderFrames(realLength)))
        }
        eval(encodedList)

        return model.rnntDecodeBatch(encodedList, seqLens: seqLens).map { tokenizer.decode($0) }
    }

    /// Encoder frames the unpadded audio would produce: mel (hop 160, win 320, center=false)
    /// then two stride-2 convolutions.
    private func encoderFrames(_ samples: Int) -> Int {
        let melFrames = max(0, (samples - MelSpectrogram.nFFT) / MelSpectrogram.hopLength + 1)
        return max(1, ((melFrames + 1) / 2 + 1) / 2)
    }
}

public struct MemoryStats: Sendable {
    public let processMemoryMB: Double
    public let gpuActiveMB: Double
    public let gpuCacheMB: Double
    public let gpuPeakMB: Double

    public static func current() -> MemoryStats {
        MemoryStats(
            processMemoryMB: residentSetSizeMB(),
            gpuActiveMB: Double(MLX.Memory.activeMemory) / 1_048_576,
            gpuCacheMB: Double(MLX.Memory.cacheMemory) / 1_048_576,
            gpuPeakMB: Double(MLX.Memory.peakMemory) / 1_048_576)
    }

    private static func residentSetSizeMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : -1
    }
}
