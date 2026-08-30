import ArgumentParser
import Foundation
import GigaAM

struct Transcribe: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Transcribe one audio/video file.")

    @Argument(help: "Path to the audio or video file.")
    var input: String

    @OptionGroup var model: ModelOptions

    @Flag(help: "Print per-segment timings instead of plain text.")
    var segments = false

    func run() throws {
        let loadStart = Date()
        let transcriber = try model.makeTranscriber()
        log("loaded \(model.modelType.rawValue) in \(elapsed(since: loadStart))")

        let start = Date()
        let result = try transcriber.transcribe(url: URL(fileURLWithPath: input))
        log("transcribed in \(elapsed(since: start))")

        if segments {
            for s in result { print(String(format: "[%.2f -> %.2f] %@", s.start, s.end, s.text)) }
        } else {
            print(result.map(\.text).joined(separator: " "))
        }
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }

    private func elapsed(since: Date) -> String {
        String(format: "%.2fs", Date().timeIntervalSince(since))
    }
}

final class InstanceCounter: @unchecked Sendable {
    private var value = 0
    private let lock = NSLock()

    func next() -> Int {
        lock.withLock {
            defer { value += 1 }
            return value
        }
    }
}

struct Bench: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "In-process throughput bench: decodes once, then loops transcription.")

    @Argument var input: String
    @OptionGroup var model: ModelOptions
    @Option(help: "Parallel worker threads.") var concurrency = 1
    @Option(help: "Total transcriptions to run.") var iterations = 16
    @Option(help: "Independent model copies; each worker gets its own.") var instances = 1

    func run() throws {
        let transcribers = try (0..<instances).map { _ in try model.makeTranscriber() }
        let transcriber = transcribers[0]

        let decodeStart = Date()
        let audio = try AudioLoader.load(url: URL(fileURLWithPath: input))
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        let audioSeconds = Double(audio.count) / 16000

        _ = transcriber.transcribe(audio: audio)  // warm kernels

        for t in transcribers.dropFirst() { _ = t.transcribe(audio: audio) }

        let group = DispatchGroup()
        let queue = DispatchQueue(label: "bench", attributes: .concurrent)
        let slots = DispatchSemaphore(value: concurrency)
        let counter = InstanceCounter()
        let start = Date()
        for _ in 0..<iterations {
            slots.wait()
            queue.async(group: group) {
                _ = transcribers[counter.next() % transcribers.count].transcribe(audio: audio)
                slots.signal()
            }
        }
        group.wait()
        let wall = Date().timeIntervalSince(start)

        print(String(format: "audio        %.1fs", audioSeconds))
        print(
            String(
                format: "decode       %.2fs (%.0fx realtime)", decodeSeconds,
                audioSeconds / decodeSeconds))
        print(
            String(
                format: "iterations   %d @ concurrency %d, %d instance(s)", iterations, concurrency,
                instances))
        print(String(format: "wall         %.2fs", wall))
        print(
            String(
                format: "throughput   %.3f rps, %.1fx realtime", Double(iterations) / wall,
                Double(iterations) * audioSeconds / wall))
    }
}
