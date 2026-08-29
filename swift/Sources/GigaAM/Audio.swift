// Audio decoding to 16 kHz mono float32. AVFoundation handles wav/mp3/m4a/aac/flac/caf
// natively; ffmpeg is the fallback for containers it refuses (webm/ogg/opus, video).

import AVFoundation
import Foundation

public enum AudioError: Error, CustomStringConvertible {
    case decodeFailed(String)

    public var description: String {
        switch self {
        case .decodeFailed(let m): return "audio decode failed: \(m)"
        }
    }
}

/// Pulls PCM frames off an AVAudioFile for AVAudioConverter's pull block.
private final class FrameReader: @unchecked Sendable {
    private let file: AVAudioFile
    private let format: AVAudioFormat
    private let capacity: AVAudioFrameCount
    private var drained = false

    init(file: AVAudioFile, format: AVAudioFormat, capacity: AVAudioFrameCount) {
        self.file = file
        self.format = format
        self.capacity = capacity
    }

    func next() -> AVAudioPCMBuffer? {
        guard !drained, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity),
            (try? file.read(into: buffer)) != nil, buffer.frameLength > 0
        else {
            drained = true
            return nil
        }
        return buffer
    }
}

public enum AudioLoader {
    public static let sampleRate: Double = 16000

    /// Decodes an upload without ever touching the disk.
    ///
    /// The obvious path - write the body to a temp file and hand it to AVAudioFile - costs
    /// a disk write per request, and a meeting is a steady stream of chunks. WAV at 16 kHz
    /// mono is parsed in place; anything else goes through ffmpeg over a pipe.
    public static func load(bytes: [UInt8]) throws -> [Float] {
        if let samples = decodeWav16kMono(bytes) { return samples }
        return try loadFFmpeg(bytes: bytes)
    }

    public static func load(url: URL) throws -> [Float] {
        do {
            return try loadNative(url: url)
        } catch {
            return try loadFFmpeg(url: url, nativeError: error)
        }
    }

    static func loadNative(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw AudioError.decodeFailed("no converter for \(inFormat)") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        let inCapacity: AVAudioFrameCount = 32768
        let outCapacity =
            AVAudioFrameCount(Double(inCapacity) * sampleRate / inFormat.sampleRate) + 1024
        var samples: [Float] = []
        let reader = FrameReader(file: file, format: inFormat, capacity: inCapacity)

        while true {
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity)
            else { throw AudioError.decodeFailed("output buffer allocation") }

            var convError: NSError?
            let status = converter.convert(to: outBuf, error: &convError) { _, outStatus in
                guard let inBuf = reader.next() else {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return inBuf
            }
            if let convError { throw convError }
            if let ch = outBuf.floatChannelData, outBuf.frameLength > 0 {
                samples.append(
                    contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(outBuf.frameLength)))
            }
            if status == .endOfStream || status == .error { break }
        }
        return samples
    }

    static func loadFFmpeg(url: URL, nativeError: Error) throws -> [Float] {
        try runFFmpeg(input: ["-i", url.path], stdin: nil, context: "\(nativeError)")
    }

    static func loadFFmpeg(bytes: [UInt8]) throws -> [Float] {
        try runFFmpeg(input: ["-i", "pipe:0"], stdin: Data(bytes), context: "in-memory decode")
    }

    private static func runFFmpeg(input: [String], stdin: Data?, context: String) throws -> [Float]
    {
        guard
            let ffmpeg = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
                .first(where: { FileManager.default.isExecutableFile(atPath: $0) })
        else { throw AudioError.decodeFailed("\(context); ffmpeg not installed") }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg)
        process.arguments =
            ["-nostdin", "-threads", "0"] + input
            + ["-f", "f32le", "-ac", "1", "-ar", String(Int(sampleRate)), "-"]

        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice

        let input = Pipe()
        if stdin != nil {
            process.standardInput = input
        }
        try process.run()

        // Feed and drain concurrently: a big upload fills the pipe buffer, and ffmpeg
        // blocks writing output until we read it - writing everything first would deadlock.
        if let stdin {
            DispatchQueue.global(qos: .userInitiated).async {
                input.fileHandleForWriting.write(stdin)
                try? input.fileHandleForWriting.close()
            }
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        guard process.terminationStatus == 0, !data.isEmpty else {
            throw AudioError.decodeFailed("\(context); ffmpeg exit \(process.terminationStatus)")
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Fast path for the format the pipeline actually speaks: RIFF/WAVE, PCM16, mono,
    /// 16 kHz. Anything else returns nil and goes to ffmpeg.
    static func decodeWav16kMono(_ bytes: [UInt8]) -> [Float]? {
        guard bytes.count > 44,
            bytes[0] == 0x52, bytes[1] == 0x49, bytes[2] == 0x46, bytes[3] == 0x46,  // RIFF
            bytes[8] == 0x57, bytes[9] == 0x41, bytes[10] == 0x56, bytes[11] == 0x45  // WAVE
        else { return nil }

        func u16(_ at: Int) -> Int { Int(bytes[at]) | Int(bytes[at + 1]) << 8 }
        // Через UInt32: размер 0xFFFFFFFF у потокового WAV в Int со знаком стал бы
        // отрицательным, и файл молча ушёл бы в ffmpeg.
        func u32(_ at: Int) -> Int {
            Int(
                UInt32(bytes[at]) | UInt32(bytes[at + 1]) << 8 | UInt32(bytes[at + 2]) << 16
                    | UInt32(bytes[at + 3]) << 24)
        }

        var offset = 12
        var dataRange: Range<Int>?
        var formatOK = false

        while offset + 8 <= bytes.count {
            let body = offset + 8
            // Обрезаем по остатку буфера, а не выходим: у потокового WAV размер поставлен
            // заведомо больше файла, и выход терял бы data-чанк целиком.
            let size = min(u32(offset + 4), bytes.count - body)

            let id = (bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
            if id == (0x66, 0x6D, 0x74, 0x20), size >= 16 {  // "fmt "
                formatOK =
                    u16(body) == 1 && u16(body + 2) == 1 && u32(body + 4) == 16000
                    && u16(body + 14) == 16
            } else if id == (0x64, 0x61, 0x74, 0x61) {  // "data"
                dataRange = body..<min(body + size, bytes.count)
                break
            }
            offset = body + size + (size % 2)
            guard offset > body - 8 else { break }
        }

        guard formatOK, let dataRange, dataRange.count >= 2 else { return nil }

        let count = dataRange.count / 2
        var samples = [Float](repeating: 0, count: count)
        bytes.withUnsafeBufferPointer { buffer in
            // Внутри withUnsafeBufferPointer у непустого массива baseAddress не nil.
            // swiftlint:disable:next force_unwrapping
            let base = buffer.baseAddress! + dataRange.lowerBound
            for i in 0..<count {
                let raw = UInt16(base[i * 2]) | (UInt16(base[i * 2 + 1]) << 8)
                samples[i] = Float(Int16(bitPattern: raw)) / 32768
            }
        }
        return samples
    }
}

/// Splits audio at low-energy points into chunks of at most `maxChunkSec`.
/// Port of gigaam_mlx.audio.split_audio.
public func splitAudio(
    _ audio: [Float], maxChunkSec: Double = 20.0, sampleRate: Int = 16000
) -> [(startSample: Int, endSample: Int)] {
    let chunkSamples = Int(maxChunkSec * Double(sampleRate))
    let minSilence = Int(0.3 * Double(sampleRate))
    var chunks: [(Int, Int)] = []
    var start = 0

    while start < audio.count {
        var end = min(start + chunkSamples, audio.count)
        if end < audio.count {
            let searchStart = start + chunkSamples / 2
            let windowLen = end - searchStart
            if windowLen > minSilence {
                // Sliding-window mean of |x| — argmin over the same positions np.convolve('valid') covers.
                var sum: Float = 0
                for i in 0..<minSilence { sum += abs(audio[searchStart + i]) }
                var best = 0
                var bestSum = sum
                for i in 1...(windowLen - minSilence) {
                    sum +=
                        abs(audio[searchStart + i + minSilence - 1])
                        - abs(audio[searchStart + i - 1])
                    if sum < bestSum {
                        bestSum = sum
                        best = i
                    }
                }
                end = searchStart + best + minSilence / 2
            }
        }
        chunks.append((start, end))
        start = end
    }
    return chunks
}
