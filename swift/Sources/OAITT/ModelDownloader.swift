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

@Observable
final class ModelDownloader: @unchecked Sendable {
    private(set) var progress: [String: Double] = [:]
    private(set) var errors: [String: String] = [:]

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
        progress[model.id] = 0
        errors[model.id] = nil

        Task { [weak self] in
            let base = URL(fileURLWithPath: directory).appendingPathComponent(model.directory)
            do {
                try FileManager.default.createDirectory(
                    at: base, withIntermediateDirectories: true)

                for (index, file) in RemoteModel.files.enumerated() {
                    guard let source = model.url(for: file) else { continue }
                    let (temporary, _) = try await URLSession.shared.download(from: source)
                    let target = base.appendingPathComponent(file)
                    try? FileManager.default.removeItem(at: target)
                    try FileManager.default.moveItem(at: temporary, to: target)
                    await self?.report(
                        model.id, Double(index + 1) / Double(RemoteModel.files.count))
                }
                await self?.finish(model.id, error: nil)
            } catch {
                await self?.finish(model.id, error: String(describing: error))
            }
        }
    }

    @MainActor
    private func report(_ id: String, _ value: Double) {
        progress[id] = value
    }

    @MainActor
    private func finish(_ id: String, error: String?) {
        progress[id] = nil
        errors[id] = error
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
                        ProgressView(value: value).frame(width: 80)
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
