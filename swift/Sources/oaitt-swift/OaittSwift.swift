import ArgumentParser
import Foundation
import GigaAM
import Hummingbird

@main
struct OaittSwift: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "oaitt-swift",
        abstract: "Native Swift/MLX GigaAM transcription server.",
        subcommands: [Transcribe.self, Serve.self, Bench.self, Balance.self],
        defaultSubcommand: Serve.self)
}

extension GigaAMModelType: ExpressibleByArgument {}

struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Directory holding <type>/weights.safetensors + tokenizer.model.")
    var modelCacheDir = "data/gigaam_mlx"

    @Option(name: .long, help: "Model variant: ctc or rnnt.")
    var modelType: GigaAMModelType = .ctc

    @Flag(
        inversion: .prefixedNo,
        help: "Run chunks without a model lock so parallel requests overlap.")
    var lockFree = true

    @Option(help: "Round chunk length up to this many seconds before encoding; 0 disables.")
    var padBucketSec = 1.0

    /// Chosen per request by the `model` field; loaded on first use.
    @Option(help: "Extra models to serve, comma separated: gigaam-ctc, gigaam-rnnt.")
    var models = ""

    @Option(help: "Drop model weights after this many idle seconds; 0 keeps them loaded.")
    var idleTimeout = 0.0

    @Option(help: "Transcripts kept keyed by audio hash; 0 disables the cache.")
    var cacheSize = 256

    @Option(help: "Seconds a cached transcript stays valid.")
    var cacheTtl = 300.0

    @Flag(inversion: .prefixedNo, help: "Record request statistics into a local SQLite file.")
    var telemetry = true

    @Flag(help: "Also keep audio and its transcript, to build a training set.")
    var dataset = false

    @Option(help: "Ceiling for the kept audio, in gigabytes.")
    var datasetLimitGb = 10.0

    @Option(help: "Where oaitt.sqlite, samples/ and logs/ live.")
    var telemetryDir = ""

    @Option(help: "Days of request logs to keep; 0 disables logging.")
    var logRetentionDays = 7

    @Flag(help: "Exit when the parent process is gone; used by the app supervisor.")
    var exitWithParent = false

    func makeTranscriber() throws -> GigaAMTranscriber {
        let dir = URL(fileURLWithPath: modelCacheDir).appendingPathComponent(modelType.rawValue)
        return try GigaAMTranscriber(
            modelDir: dir, modelType: modelType, lockFree: lockFree, padBucketSec: padBucketSec,
            idleTimeout: idleTimeout)
    }

    var storageDirectory: URL {
        telemetryDir.isEmpty
            ? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("OAITT")
            : URL(fileURLWithPath: telemetryDir)
    }

    func makeRequestLog() -> RequestLog? {
        guard logRetentionDays > 0 else { return nil }
        return RequestLog(directory: storageDirectory, retentionDays: logRetentionDays)
    }

    func makeTelemetry() -> Telemetry? {
        let directory =
            telemetryDir.isEmpty
            ? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("OAITT")
            : URL(fileURLWithPath: telemetryDir)

        return Telemetry(
            config: TelemetryConfig(
                enabled: telemetry, datasetEnabled: dataset,
                datasetLimitBytes: Int64(datasetLimitGb * 1024 * 1024 * 1024),
                directory: directory))
    }

    func makeRegistry(fallback: GigaAMTranscriber) -> ModelRegistry {
        ModelRegistry(
            names: models.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces).lowercased()
            },
            fallback: fallback, modelCacheDir: URL(fileURLWithPath: modelCacheDir),
            lockFree: lockFree, padBucketSec: padBucketSec, idleTimeout: idleTimeout)
    }
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Serve an OpenAI-compatible /v1/audio/transcriptions endpoint.")

    @Option var host = "127.0.0.1"
    @Option var port = 9007
    /// Same default as the Python service (AUTH_TOKEN=key); empty disables auth.
    @Option(help: "Bearer token clients must send; empty disables auth.")
    var apiKey = "key"

    @Flag(help: "Drop transcripts whose character rate says the model looped.")
    var confidenceFilter = false

    @OptionGroup var model: ModelOptions

    func run() async throws {
        if model.exitWithParent { Self.exitWhenOrphaned() }

        let transcriber = try model.makeTranscriber()
        let registry = model.makeRegistry(fallback: transcriber)
        let modelType = model.modelType
        let token = apiKey
        let confidenceFilter = confidenceFilter
        let cache = ResultCache(maxEntries: model.cacheSize, ttl: model.cacheTtl)
        let telemetry = model.makeTelemetry()
        let requestLog = model.makeRequestLog()
        let queue = DispatchQueue(label: "oaitt.transcribe", attributes: .concurrent)

        let router = Router()

        Self.addRoutes(
            to: router,
            context: Context(
                transcriber: transcriber, registry: registry, cache: cache, telemetry: telemetry,
                requestLog: requestLog, queue: queue, modelType: modelType, token: token,
                confidenceFilter: confidenceFilter))

        let app = Application(
            router: router,
            server: .http1(
                configuration: .init(additionalChannelHandlers: [ExpectContinueHandler()])),
            configuration: .init(address: .hostname(host, port: port), serverName: "oaitt-swift"))
        try await app.runService()
    }

    /// Everything the routes need, assembled once at startup.
    struct Context {
        let transcriber: GigaAMTranscriber
        let registry: ModelRegistry
        let cache: ResultCache
        let telemetry: Telemetry?
        let requestLog: RequestLog?
        let queue: DispatchQueue
        let modelType: GigaAMModelType
        let token: String
        let confidenceFilter: Bool
    }

    private static func addHealthRoutes(
        to router: Router<BasicRequestContext>, context: Context
    ) {
        let transcriber = context.transcriber
        let cache = context.cache
        let telemetry = context.telemetry
        let modelType = context.modelType

        router.get("/health") { _, _ -> Response in
            let memory = MemoryStats.current()
            let payload: [String: Any] = [
                "status": "healthy",
                "engine": "gigaam_mlx_swift",
                "model_type": modelType.rawValue,
                "lock_free": transcriber.lockFree,
                "memory": [
                    "process_memory_mb": round(memory.processMemoryMB * 10) / 10,
                    "gpu_memory_mb": round(memory.gpuActiveMB * 10) / 10,
                    "gpu_memory_cache_mb": round(memory.gpuCacheMB * 10) / 10,
                    "gpu_memory_peak_mb": round(memory.gpuPeakMB * 10) / 10,
                ],
            ]
            let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
            return Response(
                status: .ok, headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(bytes: data)))
        }

        router.get("/health/detailed") { _, _ -> Response in
            let memory = MemoryStats.current()
            var payload: [String: Any] = [
                "status": "healthy",
                "engine": "gigaam_mlx_swift",
                "formats": ["json", "text", "srt", "vtt", "tsv", "verbose_json"],
            ]
            payload["model"] =
                [
                    "type": modelType.rawValue,
                    "loaded": transcriber.isLoaded,
                    "idle_timeout": transcriber.idleTimeout,
                    "lock_free": transcriber.lockFree,
                    "pad_bucket_sec": transcriber.padBucketSec,
                ] as [String: Any]
            payload["memory"] =
                [
                    "process_memory_mb": round(memory.processMemoryMB * 10) / 10,
                    "gpu_memory_mb": round(memory.gpuActiveMB * 10) / 10,
                    "gpu_memory_cache_mb": round(memory.gpuCacheMB * 10) / 10,
                    "gpu_memory_peak_mb": round(memory.gpuPeakMB * 10) / 10,
                ] as [String: Any]
            payload["cache"] = cache.stats
            payload["telemetry"] = telemetry?.summary(sinceDays: 30) ?? ["enabled": false]
            return Self.json(payload)
        }

    }

    /// Route wiring lives apart from startup so neither half grows past reading size.
    static func addRoutes(to router: Router<BasicRequestContext>, context: Context) {
        let (transcriber, registry, cache) = (context.transcriber, context.registry, context.cache)
        let (telemetry, queue, modelType) = (context.telemetry, context.queue, context.modelType)
        let requestLog = context.requestLog
        let (token, confidenceFilter) = (context.token, context.confidenceFilter)
        addHealthRoutes(to: router, context: context)

        router.post("/asr") { request, _ -> Response in
            if !token.isEmpty, request.headers[.authorization] != "Bearer \(token)" {
                throw HTTPError(.unauthorized)
            }
            let query = request.uri.queryParameters
            let format = query["output"].map(String.init) ?? "json"
            let language = query["language"].map(String.init) ?? "ru"
            let modelName = query["model"].map(String.init)

            let (segments, seconds) = try await Self.runTranscription(
                request: request, queue: queue,
                transcriber: registry.transcriber(for: modelName), cache: cache,
                telemetry: telemetry, requestLog: requestLog)
            return Self.render(
                segments, format: format, audioSeconds: seconds, language: language,
                confidenceFilter: confidenceFilter)
        }

        router.post("/v1/audio/transcriptions") { request, _ -> Response in
            if !token.isEmpty, request.headers[.authorization] != "Bearer \(token)" {
                throw HTTPError(.unauthorized)
            }
            guard let contentType = request.headers[.contentType],
                let boundary = Multipart.boundary(fromContentType: contentType)
            else { throw HTTPError(.badRequest, message: "expected multipart/form-data") }

            let buffer = try await request.body.collect(upTo: 512 * 1024 * 1024)
            let parts = Multipart.parse(Array(buffer.readableBytesView), boundary: boundary)
            func field(_ name: String) -> String? {
                parts.first { $0.name == name }.map { String(decoding: $0.body, as: UTF8.self) }
            }
            let responseFormat = field("response_format") ?? "json"
            let language = field("language") ?? "ru"

            let (segments, audioSeconds) = try await Self.runTranscription(
                parts: parts, queue: queue,
                transcriber: registry.transcriber(for: field("model")), cache: cache,
                telemetry: telemetry, requestLog: requestLog)
            return Self.render(
                segments, format: responseFormat, audioSeconds: audioSeconds,
                language: language, confidenceFilter: confidenceFilter)
        }

    }

    /// Reads the multipart file part, decodes it and runs the model off the event loop.
    static func runTranscription(
        request: Request, queue: DispatchQueue, transcriber: any ASREngine, cache: ResultCache,
        telemetry: Telemetry?, requestLog: RequestLog?
    ) async throws -> ([Segment], Double) {
        guard let contentType = request.headers[.contentType],
            let boundary = Multipart.boundary(fromContentType: contentType)
        else { throw HTTPError(.badRequest, message: "expected multipart/form-data") }

        let buffer = try await request.body.collect(upTo: 512 * 1024 * 1024)
        return try await runTranscription(
            parts: Multipart.parse(Array(buffer.readableBytesView), boundary: boundary),
            queue: queue, transcriber: transcriber, cache: cache, telemetry: telemetry,
            requestLog: requestLog)
    }

    static func runTranscription(
        parts: [MultipartPart], queue: DispatchQueue, transcriber: any ASREngine,
        cache: ResultCache, telemetry: Telemetry?, requestLog: RequestLog?
    ) async throws -> ([Segment], Double) {
        guard let filePart = parts.first(where: { $0.name == "file" || $0.name == "audio_file" })
        else { throw HTTPError(.badRequest, message: "missing 'file' field") }

        let key = ResultCache.key(
            bytes: filePart.body, parameters: ["model": transcriber.name])
        let started = Date()
        if let cached = cache.get(key) {
            telemetry?.recordRequest(
                model: transcriber.name, audioSeconds: cached.audioSeconds,
                durationMs: Date().timeIntervalSince(started) * 1000, status: 200, cached: true,
                rssMB: MemoryStats.current().processMemoryMB,
                gpuMB: MemoryStats.current().gpuActiveMB)
            requestLog?.record(
                model: transcriber.name, audioSeconds: cached.audioSeconds,
                durationMs: Date().timeIntervalSince(started) * 1000, status: 200, cached: true,
                text: cached.segments.map(\.text).joined(separator: " "))
            return cached
        }

        // No temp file: a meeting is a steady stream of chunks, and a disk write per
        // request wears the SSD for nothing.
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(
                    with: Result {
                        let audio = try AudioLoader.load(bytes: filePart.body)
                        let result = (
                            segments: transcriber.transcribe(audio: audio),
                            audioSeconds: Double(audio.count) / AudioLoader.sampleRate
                        )
                        cache.put(key, result)

                        let memory = MemoryStats.current()
                        telemetry?.recordRequest(
                            model: transcriber.name, audioSeconds: result.audioSeconds,
                            durationMs: Date().timeIntervalSince(started) * 1000, status: 200,
                            cached: false, rssMB: memory.processMemoryMB,
                            gpuMB: memory.gpuActiveMB)
                        requestLog?.record(
                            model: transcriber.name, audioSeconds: result.audioSeconds,
                            durationMs: Date().timeIntervalSince(started) * 1000, status: 200,
                            cached: false,
                            text: result.segments.map(\.text).joined(separator: " "))
                        telemetry?.recordSample(
                            model: transcriber.name, audio: filePart.body,
                            audioFormat: (filePart.filename as NSString?)?.pathExtension ?? "wav",
                            text: result.segments.map(\.text).joined(separator: " "))
                        return result
                    })
            }
        }
    }

    static func render(
        _ segments: [Segment], format: String, audioSeconds: Double, language: String,
        confidenceFilter: Bool
    ) -> Response {
        let text = segments.map(\.text).joined(separator: " ")
        let confidence = ConfidenceMetrics.evaluate(text: text, audioSeconds: audioSeconds)

        // A wildly high character rate means the model looped; drop the text like the
        // Python route does (src/routes/openai.py:221) rather than return garbage.
        let filtered = confidenceFilter && !confidence.isReliable
        let outSegments = filtered ? [] : segments
        let outText = filtered ? "" : text

        switch format {
        case "text":
            return plain(outText, type: "text/plain; charset=utf-8")
        case "srt":
            return plain(Formatters.srt(outSegments), type: "text/plain; charset=utf-8")
        case "vtt":
            return plain(Formatters.vtt(outSegments), type: "text/vtt; charset=utf-8")
        case "tsv":
            return plain(Formatters.tsv(outSegments), type: "text/plain; charset=utf-8")
        case "verbose_json":
            var payload: [String: Any] = [
                "text": outText,
                "task": "transcribe",
                "language": language,
                "duration": audioSeconds,
                "segments": outSegments.enumerated().map { index, s in
                    ["id": index, "start": s.start, "end": s.end, "text": s.text] as [String: Any]
                },
                "confidence": confidence.json,
            ]
            if let rate = confidence.charsPerSecond { payload["chars_per_second"] = rate }
            return json(payload)
        default:
            return json(["text": outText])
        }
    }

    /// Иначе воркер переживает смерть приложения, остаётся сиротой и держит порт: новый
    /// запуск не может его занять и выглядит как "worker crashed".
    static func exitWhenOrphaned() {
        let parent = getppid()
        Thread.detachNewThread {
            while true {
                Thread.sleep(forTimeInterval: 2)
                if getppid() != parent || getppid() == 1 { Foundation.exit(0) }
            }
        }
    }

    static func plain(_ body: String, type: String) -> Response {
        Response(
            status: .ok, headers: [.contentType: type],
            body: .init(byteBuffer: ByteBuffer(string: body)))
    }

    static func json(_ payload: [String: Any]) -> Response {
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return Response(
            status: .ok, headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data)))
    }
}
