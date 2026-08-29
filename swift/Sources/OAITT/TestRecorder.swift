import AVFoundation
import Observation
import SwiftUI

/// Запись с микрофона и отправка в собственный эндпоинт.
///
/// Нужна, чтобы проверить работоспособность целиком - микрофон, сервис, модель - не
/// открывая терминал. Пишем сразу в 16 kHz mono PCM16: это ровно тот формат, который
/// движок разбирает в памяти, без ffmpeg и без пересжатия.
@MainActor
@Observable
final class TestRecorder: NSObject {
    enum State: Equatable {
        case idle
        case recording
        case transcribing
        case done(String)
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var level: Float = 0

    private var recorder: AVAudioRecorder?
    private var meterTimer: Timer?
    private var fileURL: URL?

    var isBusy: Bool { state == .recording || state == .transcribing }

    func toggle(port: Int, apiKey: String) {
        if state == .recording {
            stopAndSend(port: port, apiKey: apiKey)
        } else {
            start()
        }
    }

    private func start() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("oaitt-test-\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]

        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.isMeteringEnabled = true
            guard recorder.record() else {
                state = .failed("microphone is not available")
                return
            }
            self.recorder = recorder
            fileURL = url
            state = .recording

            meterTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
                Task { @MainActor in self.updateLevel() }
            }
        } catch {
            state = .failed(String(describing: error))
        }
    }

    private func updateLevel() {
        guard let recorder, recorder.isRecording else { return }
        recorder.updateMeters()
        // Пик в дБ (-160...0) в 0...1, ниже -50 дБ считаем тишиной.
        let decibels = recorder.averagePower(forChannel: 0)
        level = max(0, min(1, (decibels + 50) / 50))
    }

    private func stopAndSend(port: Int, apiKey: String) {
        meterTimer?.invalidate()
        meterTimer = nil
        recorder?.stop()
        recorder = nil
        level = 0

        guard let fileURL, let data = try? Data(contentsOf: fileURL) else {
            state = .failed("recording is empty")
            return
        }
        try? FileManager.default.removeItem(at: fileURL)
        self.fileURL = nil

        // 44 байта - это только WAV-заголовок, значит не записалось ничего.
        guard data.count > 44 + 16000 else {
            state = .failed("too short, hold the button while speaking")
            return
        }

        state = .transcribing
        Task { await send(data, port: port, apiKey: apiKey) }
    }

    private func send(_ audio: Data, port: Int, apiKey: String) async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/v1/audio/transcriptions") else {
            return
        }
        let boundary = UUID().uuidString
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue(
            "multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        var body = Data()
        func field(_ name: String, _ value: String) {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data("\(value)\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(
            Data(
                "Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(audio)
        body.append(Data("\r\n".utf8))
        field("response_format", "text")
        body.append(Data("--\(boundary)--\r\n".utf8))
        request.httpBody = body

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let text = String(decoding: data, as: UTF8.self).trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard code == 200 else {
                state = .failed("HTTP \(code): \(text.prefix(120))")
                return
            }
            state = .done(text.isEmpty ? "(silence)" : text)
        } catch {
            state = .failed(String(describing: error))
        }
    }

    func reset() {
        state = .idle
    }
}

struct TestRecorderView: View {
    var settings: AppSettings
    @State private var recorder = TestRecorder()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Button {
                    recorder.toggle(port: settings.port, apiKey: settings.apiKey)
                } label: {
                    Label(
                        recorder.state == .recording ? "Stop and transcribe" : "Test microphone",
                        systemImage: recorder.state == .recording ? "stop.circle.fill" : "mic")
                }
                .disabled(recorder.state == .transcribing)

                if recorder.state == .recording {
                    // Уровень видно сразу: если полоса не шевелится, дело в микрофоне,
                    // а не в распознавании.
                    ProgressView(value: Double(recorder.level))
                        .frame(width: 60)
                        .tint(recorder.level > 0.05 ? .green : .secondary)
                } else if recorder.state == .transcribing {
                    ProgressView().controlSize(.small)
                }
            }

            switch recorder.state {
            case .done(let text):
                Text(verbatim: text)
                    .font(.caption)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            case .failed(let message):
                Text(verbatim: message)
                    .font(.caption).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            default:
                EmptyView()
            }
        }
    }
}
