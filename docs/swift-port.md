# Нативный Swift/MLX порт

Код - [`swift/`](../swift), замеры - [benchmarks.md](benchmarks.md).

## Зачем

Проверка гипотезы, что нативная реализация на MLX Swift даст меньший расход памяти и более
предсказуемую latency, чем Python-сервис. Гипотеза подтвердилась частично: в один поток
Swift быстрее (CTC 1.8x, RNNT 1.35x), unified memory тратит в 2.2x меньше, хвосты latency
ровнее в разы - но внутри процесса не масштабируется совсем.

## Устройство

```
swift/
  Package.swift                        mlx-swift 0.31.6, Hummingbird 2.26, ArgumentParser
  Sources/GigaAM/
    Model.swift        Conformer encoder + RoPE-attention + CTC head + RNNT decoder/joint
    Mel.swift          log-mel через MLXFFT, htk-фильтры (порт librosa)
    Audio.swift        AVFoundation -> 16k mono, ffmpeg fallback; splitAudio
    Tokenizer.swift    SentencePiece decode парсингом protobuf, без зависимостей
    Transcriber.swift  чанкинг, bucket-padding, батч-декод RNNT, MemoryStats
    Formatters.swift   srt / vtt / tsv, ConfidenceMetrics
  Sources/oaitt-swift/
    OaittSwift.swift   CLI: transcribe / serve / bench
    Multipart.swift    multipart/form-data reader
```

Веса берутся те же, что у Python-версии: `data/gigaam_mlx/<type>/weights.safetensors` +
`tokenizer.model`. Конвертация не нужна - имена параметров в Swift-модулях заданы через
`@ModuleInfo(key:)` так, чтобы совпасть с существующими ключами safetensors.

## Запуск

```bash
./run_swift_asr.sh                                  # rnnt, порт 8300
GIGAAM_MLX_MODEL_TYPE=ctc PORT=8301 ./run_swift_asr.sh
AUTH_TOKEN=secret ./run_swift_asr.sh                # с bearer-авторизацией

swift/.build/release/oaitt-swift transcribe file.ogg \
  --model-cache-dir data/gigaam_mlx --model-type ctc --segments

swift/.build/release/oaitt-swift bench file.wav \
  --model-cache-dir data/gigaam_mlx --model-type rnnt --concurrency 4 --iterations 16
```

## API

| Метод | Что |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-совместимый; `response_format`: json, text, srt, vtt, tsv, verbose_json; поля `model`, `language` |
| `GET /health` | статус, тип модели, память процесса и GPU |
| `GET /health/detailed` | плюс режим модели, `pad_bucket_sec`, статистика кеша, список форматов |
| `GET /v1/models` | модели, доступные в поле `model`, и загружены ли они |
| `POST /asr` | свой роут с query-параметрами `output`, `language`, `model` и полем `audio_file` |

`verbose_json` отдаёт сегменты и блок `confidence` с `chars_per_second`,
`chars_per_second_ratio`, `chars_per_second_threshold`, `high_char_rate`, `is_reliable` -
те же поля и та же логика порога, что в `src/routes/openai.py`. Флаг `--confidence-filter`
включает отбрасывание результата, когда символьная скорость говорит, что модель зациклилась.

## Ключевые решения

**Bucket-padding.** Длина чанка округляется вверх до `--pad-bucket-sec` (1.0s). MLX
компилирует кернелы и кеширует буферы под каждую форму тензора, поэтому сырая нарезка -
своя длина у каждого чанка - убивает и то, и другое. Дало CTC 314x -> 496x, RNNT
115x -> 142x.

**Батчевый greedy-декод RNNT.** Чанки одного запроса декодируются вместе: один
GPU->CPU sync на кадр на весь батч вместо одного на кадр на чанк. Состояние LSTM
продвигается маской `which()` только у строк, выдавших символ. Дало 142x -> 241x на
длинном файле, текст совпадает побуквенно. CTC оставлен поточным - у него один argmax, и
батч ему только мешает (был откат 508x -> 340x).

**Кеш по хешу загруженного файла.** Ключ считается по байтам тела запроса до его записи
во временный файл и декодирования, поэтому попадание не стоит ничего. Платформа ретраит
задачи, и каждый повтор приносит тот же файл: `--cache-size`, `--cache-ttl`.

**Ответ на `Expect: 100-continue`.** Hummingbird этот заголовок не обрабатывает, а curl
шлёт его для крупных загрузок и без ответа ждёт секунду перед отправкой тела. На запросе
4.4 МБ это была ровно секунда из 1.03. `ExpectContinueHandler` в пайплайне канала снимает
задержку: стало 0.03s.

**Lock-free по умолчанию.** MLX-операции ленивы и потокобезопасны на построение графа,
GPU-исполнение сериализует рантайм. `--no-lock-free` возвращает лок вокруг модели.

**Декод аудио.** AVFoundation тянет wav/mp3/m4a/aac/flac/caf нативно; для ogg/opus/webm и
видео есть fallback на ffmpeg. Ogg через ffmpeg - 0.13s на 137s аудио, не узкое место.

## Приложение

`./build-app.sh` собирает `OAITT.app` - приложение в строке меню, которое держит пул
воркеров и балансировщик перед ними. CLI и `mlx.metallib` лежат внутри бандла, веса
качаются при первом запуске.

## Ограничения

- **Не масштабируется внутри процесса.** Плато наступает при concurrency 1 и не сдвигается
  ни потоками, ни пулом моделей (4 копии дали 143x против 144x у одной). Масштабируется
  только процессами: 144 -> 263 -> 369x на 1/2/4 процессах. Причина - сериализация в
  MLX Swift; per-stream параллелизм недоступен, слои не пробрасывают `stream`, всё идёт в
  default stream, а `using(stream:)` в API нет, только `using(device:)`.
- **`mlx.metallib` копируется из питоновского venv.** SwiftPM не собирает Metal-шейдеры
  mlx-swift (это делает только их Xcode-проект), и без метallib бинарь падает с
  `Failed to load the default metallib`. `run_swift_asr.sh` берёт готовый файл из
  `venv/lib/*/site-packages/mlx/lib/mlx.metallib`; MLX ищет colocated `mlx.metallib`
  первым. Версии 0.32.2 (wheel) против 0.31.1 (mlx-swift) на практике совместимы, но при
  апгрейде сломается здесь.
- Нет: word timestamps, multilingual-моделей, пула воркеров.
