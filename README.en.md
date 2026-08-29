# OAITT - Open AI Transformer Transcriber

Speech-to-text service built on GigaAM with an OpenAI-compatible API. Two implementations
share one API: a Python service and a native Swift build for Apple Silicon, packaged as a
macOS menu bar app.

Русская версия - [`readme.md`](readme.md).

## Quick start

```bash
make            # list every command
make prepare    # environment and GigaAM MLX weights
make run        # Python service on 9007
make app-run    # macOS app
```

## macOS app

![OAITT menu](docs/images/menu.png)

The native Swift/MLX build, wrapped in a menu bar app. One download: the `.app` is about
75 MB, weights are fetched from HuggingFace on first launch.

- Serves an OpenAI-compatible endpoint on port 9007 with the token `key` - same port and
  token as the Python service, so it is a drop-in replacement.
- Runs workers as separate processes and restarts the ones that die. A single worker owns
  the port directly; several sit behind a built-in balancer.
- Shows live statistics: requests, speed, p95, a per-minute request chart, CPU and memory
  of every worker.
- Writes a request log rotated after 7 days, opened from the menu.
- Can collect audio and transcripts into a training set - off by default, capped at 10 GB.

Workers are processes on purpose: MLX does not scale inside one. A pool of model copies
there changes nothing (143x against 144x for a single copy); throughput comes from more
processes - 144x, 263x, 369x for one, two and four.

## API

Both implementations answer the same endpoints:

| Method | What |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible; `response_format`: json, text, srt, vtt, tsv, verbose_json |
| `POST /asr` | Same work, query parameters instead of form fields |
| `GET /health`, `GET /health/detailed` | Status, memory, cache and telemetry |
| `GET /v1/models` | Models available in the `model` field |

```bash
curl -X POST http://localhost:9007/v1/audio/transcriptions \
  -H "Authorization: Bearer key" \
  -F "file=@audio.ogg" \
  -F "response_format=verbose_json"
```

Set `AUTH_TOKEN` to change the token; an empty value disables authentication.

## Performance

MacBook Pro M4 Max, 137.4 s of audio, `xRT` means times faster than real time.

| concurrency | CTC Python | CTC Swift | RNNT Python | RNNT Swift |
|---|---|---|---|---|
| 1 | 271x | **496x** | 105x | **142x** |
| 4 | **602x** | 514x | **280x** | 150x |

Swift is faster on a single request and holds far steadier latency (p50 to max spread of
6% against 4.5x), and uses 2.2x less unified memory. It does not scale inside a process,
so throughput there comes from running several: 144x, 263x, 369x for one, two, four.

Full numbers, including what was measured and rejected, are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Documentation

- [`docs/INDEX.md`](docs/INDEX.md) - index
- [`docs/swift-port.md`](docs/swift-port.md) - how the Swift port works and where it does not
- [`docs/benchmarks.md`](docs/benchmarks.md) - measurements and transcription quality

## Building

```bash
make swift-build   # CLI and Metal shaders
make app           # OAITT.app
make check         # format, lint, build, tests
```

`mlx.metallib` is compiled from mlx-swift sources by `swift/build-metallib.sh` - the build
does not depend on a Python environment.

## Licence

MIT. GigaAM weights are MIT, copyright GigaChat Team. MLX Swift is MIT, copyright Apple.
