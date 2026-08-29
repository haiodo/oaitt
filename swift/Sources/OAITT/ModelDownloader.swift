// Fetches model weights from HuggingFace on first use.
//
// The weights are not shipped inside the app: CTC and RNNT are about 850 MB each, and
// together they would push the release asset close to GitHub's per-file limit. They are
// public and MIT-licensed, so downloading them on demand is both legal and smaller.

import Foundation
import Observation
import SwiftUI

struct RemoteModel: Identifiable, Sendable, Codable, Equatable {
    let id: String
    let title: String
    let repository: String
    let directory: String

    static let builtin: [RemoteModel] = [
        RemoteModel(
            id: "ctc", title: "GigaAM v3 CTC - faster",
            repository: "aystream/GigaAM-v3-e2e-ctc-mlx", directory: "ctc"),
        RemoteModel(
            id: "rnnt", title: "GigaAM v3 RNNT - more accurate",
            repository: "aystream/GigaAM-v3-e2e-rnnt-mlx", directory: "rnnt"),
    ]

    static let files = ["weights.safetensors", "tokenizer.model"]

    func url(for file: String) -> URL? {
        URL(string: "https://huggingface.co/\(repository)/resolve/main/\(file)")
    }
}

/// Качает веса с прогрессом по мере записи байтов.
///
/// `URLSession.download(from:)` отдаёт файл целиком и промежуточного прогресса не даёт: на
/// 843 МБ и типичных 3 МБ/с это пять минут неподвижной полосы, неотличимых от зависания.
/// Поэтому здесь делегат с `didWriteData`.
@Observable
final class ModelDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private(set) var progress: [String: Double] = [:]
    private(set) var speed: [String: Double] = [:]
    private(set) var errors: [String: String] = [:]

    private struct Job {
        let modelID: String
        let target: URL
        let fileIndex: Int
        let fileCount: Int
        let startedAt = Date()
    }

    @ObservationIgnored private var jobs: [Int: Job] = [:]
    @ObservationIgnored private var queues: [String: [(url: URL, target: URL)]] = [:]
    private let lock = NSLock()
    /// Сессию строим при первом обращении: делегатом выступает сам объект, а ссылаться на
    /// self в инициализаторе свойства нельзя. @Observable не переваривает lazy, поэтому
    /// вручную.
    @ObservationIgnored private var storedSession: URLSession?
    private var session: URLSession {
        lock.withLock {
            if let storedSession { return storedSession }
            let created = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
            storedSession = created
            return created
        }
    }

    func isInstalled(_ model: RemoteModel, in directory: String) -> Bool {
        let base = URL(fileURLWithPath: directory).appendingPathComponent(model.directory)
        return RemoteModel.files.allSatisfy {
            FileManager.default.fileExists(atPath: base.appendingPathComponent($0).path)
        }
    }

    func isDownloading(_ model: RemoteModel) -> Bool {
        progress[model.id] != nil
    }

    func download(_ model: RemoteModel, into directory: String) {
        guard progress[model.id] == nil else { return }

        let base = URL(fileURLWithPath: directory).appendingPathComponent(model.directory)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)

        let pending = RemoteModel.files.compactMap { file -> (url: URL, target: URL)? in
            guard let url = model.url(for: file) else { return nil }
            return (url, base.appendingPathComponent(file))
        }
        guard !pending.isEmpty else { return }

        progress[model.id] = 0
        errors[model.id] = nil
        lock.withLock { queues[model.id] = pending }
        startNext(modelID: model.id)
    }

    private func startNext(modelID: String) {
        let next: (url: URL, target: URL)? = lock.withLock {
            guard var queue = queues[modelID], !queue.isEmpty else { return nil }
            let item = queue.removeFirst()
            queues[modelID] = queue
            return item
        }

        guard let next else {
            Task { @MainActor in
                self.progress[modelID] = nil
                self.speed[modelID] = nil
            }
            return
        }

        let remaining = lock.withLock { queues[modelID]?.count ?? 0 }
        let task = session.downloadTask(with: next.url)  // session берёт лок сам, снаружи его нет
        lock.withLock {
            jobs[task.taskIdentifier] = Job(
                modelID: modelID, target: next.target,
                fileIndex: RemoteModel.files.count - remaining - 1,
                fileCount: RemoteModel.files.count)
        }
        task.resume()
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let job = lock.withLock({ jobs[downloadTask.taskIdentifier] }),
            totalBytesExpectedToWrite > 0
        else { return }

        let fileFraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        let overall = (Double(job.fileIndex) + fileFraction) / Double(job.fileCount)
        let elapsed = Date().timeIntervalSince(job.startedAt)
        let bytesPerSecond = elapsed > 0 ? Double(totalBytesWritten) / elapsed : 0

        Task { @MainActor in
            self.progress[job.modelID] = min(1, overall)
            self.speed[job.modelID] = bytesPerSecond
        }
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let job = lock.withLock({ jobs.removeValue(forKey: downloadTask.taskIdentifier) })
        else { return }

        do {
            try? FileManager.default.removeItem(at: job.target)
            try FileManager.default.moveItem(at: location, to: job.target)
            // Временный файл URLSession создаёт с правами 600, и moveItem их сохраняет -
            // общий каталог весов оказался бы нечитаемым для других пользователей.
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o644], ofItemAtPath: job.target.path)
        } catch {
            lock.withLock { queues[job.modelID] = [] }
            Task { @MainActor in
                self.errors[job.modelID] = String(describing: error)
                self.progress[job.modelID] = nil
            }
            return
        }
        startNext(modelID: job.modelID)
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?
    ) {
        guard let error, let job = lock.withLock({ jobs.removeValue(forKey: task.taskIdentifier) })
        else { return }
        lock.withLock { queues[job.modelID] = [] }
        Task { @MainActor in
            self.errors[job.modelID] = String(describing: error)
            self.progress[job.modelID] = nil
            self.speed[job.modelID] = nil
        }
    }
}

struct ModelListView: View {
    @Bindable var settings: AppSettings
    @State private var downloader = ModelDownloader()
    @State private var showingAdd = false

    private var models: [RemoteModel] {
        RemoteModel.builtin + RemoteModel.decode(settings.customModels)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(models) { model in
                HStack {
                    Image(
                        systemName: downloader.isInstalled(model, in: settings.modelCacheDir)
                            ? "checkmark.circle.fill" : "arrow.down.circle"
                    )
                    .foregroundStyle(
                        downloader.isInstalled(model, in: settings.modelCacheDir)
                            ? Color.green : Color.secondary)

                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.title).font(.caption)
                        if let error = downloader.errors[model.id] {
                            Text(error).font(.caption2).foregroundStyle(.red).lineLimit(1)
                        }
                    }

                    Spacer()

                    if let value = downloader.progress[model.id] {
                        VStack(alignment: .trailing, spacing: 1) {
                            ProgressView(value: value).frame(width: 90)
                            if let rate = downloader.speed[model.id], rate > 0 {
                                Text(
                                    verbatim: String(
                                        format: "%.0f%%  %.1f MB/s", value * 100,
                                        rate / 1024 / 1024)
                                )
                                .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    } else if !downloader.isInstalled(model, in: settings.modelCacheDir) {
                        Button("Download") {
                            downloader.download(model, into: settings.modelCacheDir)
                        }
                        .controlSize(.small)
                    }
                }
            }

            HStack {
                Button("Add model...") { showingAdd = true }.controlSize(.small)
                Spacer()
            }
        }
        .sheet(isPresented: $showingAdd) {
            AddModelSheet(settings: settings)
        }
    }
}

extension RemoteModel {
    static func decode(_ json: String) -> [RemoteModel] {
        (try? JSONDecoder().decode([RemoteModel].self, from: Data(json.utf8))) ?? []
    }

    static func encode(_ models: [RemoteModel]) -> String {
        guard let data = try? JSONEncoder().encode(models) else { return "[]" }
        return String(decoding: data, as: UTF8.self)
    }
}

/// Any GigaAM-format export works: the weights and tokenizer are loaded by name, so a
/// fine-tune of the same architecture drops straight in.
struct AddModelSheet: View {
    @Bindable var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var repository = ""
    @State private var directory = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add a model").font(.headline)
            Form {
                LabeledContent("Name") {
                    TextField("", text: $name, prompt: Text("my-gigaam"))
                        .textFieldStyle(.roundedBorder)
                }
                LabeledContent("HuggingFace repo") {
                    TextField("", text: $repository, prompt: Text("user/GigaAM-v3-e2e-ctc-mlx"))
                        .textFieldStyle(.roundedBorder)
                }
                LabeledContent("Subfolder") {
                    TextField("", text: $directory, prompt: Text("ctc"))
                        .textFieldStyle(.roundedBorder)
                }
            }
            .formStyle(.columns)

            Text(
                "Expects weights.safetensors and tokenizer.model in the repo root - "
                    + "the GigaAM MLX layout."
            )
            .font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add") {
                    var models = RemoteModel.decode(settings.customModels)
                    models.append(
                        RemoteModel(
                            id: name, title: name, repository: repository,
                            directory: directory.isEmpty ? name : directory))
                    settings.customModels = RemoteModel.encode(models)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(name.isEmpty || repository.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}

/// Which models the server answers for, beyond the default one. A checkbox per model
/// beats a comma separated field: the names are fixed and the user should not have to
/// remember their spelling.
struct ServedModelsView: View {
    @Bindable var settings: AppSettings

    private var models: [RemoteModel] {
        RemoteModel.builtin + RemoteModel.decode(settings.customModels)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Also serve").font(.callout)
            ForEach(models) { model in
                Toggle(
                    "gigaam-\(model.id)",
                    isOn: Binding(
                        get: { settings.isServed("gigaam-\(model.id)") },
                        set: { settings.setServed("gigaam-\(model.id)", $0) })
                )
                .toggleStyle(.checkbox)
                .font(.callout)
            }
            Text("Clients pick these by the model field in the request. Loaded on first use.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
