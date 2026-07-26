# OAITT — Open AI Transformer Transcriber

**Высокопроизводительный сервис распознавания речи с поддержкой Whisper и GigaAM**

OAITT — это speech-to-text сервис транскрипции с поддержкой нескольких ASR движков, OpenAI-совместимым API и расширенными метриками качества.

---

## Возможности

- **Шесть ASR движков**: GigaAM Native, GigaAM MLX, GigaAM Multilingual MLX (Apple Silicon), GigaAM via HF, Hugging Face Transformers и WhisperX
- **70+ языков**: GigaAM Multilingual - лучший WER на русском, казахском, киргизском, узбекском
- **OpenAI-совместимый API**: Drop-in замена для OpenAI Whisper API
- **Точные временные метки**: На уровне слов (с WhisperX)
- **Метрики уверенности**: Оценка качества транскрипции
- **Адаптивные таймауты**: Защита от зависания модели
- **Apple Silicon**: Поддержка MPS для Mac
- **Множество форматов**: JSON, текст, SRT, VTT, TSV
- **Высокая производительность**: До 327x realtime с GigaAM Native на Apple Silicon

---

## Производительность

### Тестовая конфигурация

| Параметр | Значение |
|----------|----------|
| **Модель** | MacBook Pro (Mac16,5) |
| **Чип** | Apple M4 Max |
| **Ядра** | 16 (12 производительных + 4 энергоэффективных) |
| **Память** | 48 GB |

### Результаты бенчмарка

Тестовое аудио: **137.4 секунд**, режим: full. На каждый движок — прогрев (в замер не идёт),
затем итерации до заполнения бюджета в 60 секунд, но не более 60 прогонов.

| Движок | Avg (s) | Min (s) | Max (s) | Скорость | RSS | GPU | Пункт. |
|--------|---------|---------|---------|----------|-----|-----|--------|
| **GigaAM Native (CTC)** | **0.43** | 0.42 | 0.48 | **320.07x** | 1445 MB | 1116 MB | да |
| GigaAM Multilingual (CTC) | 0.61 | 0.45 | 0.76 | 226.08x | 682 MB | 1116 MB | нет |
| GigaAM MLX (CTC) | 0.72 | 0.63 | 1.01 | 191.05x | 1525 MB | 843 MB | да |
| GigaAM Multilingual Large MLX (int8) | 0.78 | 0.58 | 0.93 | 176.54x | 1274 MB | 666 MB | нет |
| GigaAM Multilingual Large MLX (fp16) | 0.82 | 0.58 | 1.16 | 167.50x | 1710 MB | 1116 MB | нет |
| GigaAM Multilingual Large (CTC) | 1.13 | 0.74 | 1.54 | 121.77x | 303 MB | 3232 MB | нет |
| GigaAM MLX (RNNT) | 1.47 | 1.42 | 1.57 | 93.37x | 1519 MB | 849 MB | да |
| GigaAM Native (RNNT) | 3.31 | 3.12 | 3.45 | 41.45x | 1586 MB | 1116 MB | да |

Воспроизвести: `./benchmark.sh` — прогонит все движки и откроет HTML-отчёт.

> **GigaAM Native (CTC) — самый быстрый вариант** для русского языка (320x realtime),
> использует прямую передачу тензоров без временных файлов.
>
> **MLX даёт заметный выигрыш на больших моделях.** Multilingual Large (600M) в MLX — 177x
> против 122x у PyTorch-версии той же модели. На v3 (240M) картина обратная: PyTorch-CTC
> быстрее MLX-CTC (320x против 191x), а вот RNNT в MLX вдвое быстрее (93x против 41x).
>
> **RSS** — память процесса по данным `/health`, **GPU** — unified memory,
> **Пункт.** — модель проставляет пунктуацию (charwise CTC этого не делают).
>
> Замер на одном файле. В проде MLX-движки держат больше: кэш аллокатора растёт с числом
> уникальных длин чанков после VAD-split, а в lock-free режиме каждый параллельный клиент
> держит свои активации. RSS у PyTorch-multilingual занижен — `/health` опрашивается после
> прогона, когда часть страниц уже вытеснена.

**Не в наборе по умолчанию:**

| Движок | Причина |
|--------|---------|
| GigaAM (Transformers) | Кастомный код `ai-sage/GigaAM-v3` несовместим с transformers 5.x (`.item()` на meta-тензоре). Используйте native — та же модель |
| Whisper Large V3 | ~9.5x realtime, на порядок медленнее GigaAM при сопоставимом качестве на русском |
| WhisperX Large V3 | Зависает на длинном аудио (сервер стартует, запрос не возвращается) |

Запустить явно: `./benchmark.sh -s run_whisper_large_v3.sh`

---

## Docker

Быстрый запуск через Docker (CPU режим):

```bash
# 1. Подготовка (один раз)
./prepare.sh # Скачает модели GigaAM

# 2. Сборка образа (локально)
./build.sh myuser/oaitt-gigaam 1.0.0

# 3. Или сборка + публикация для Linux AMD64/ARM64
./build.sh --amd64 --arm64 --push myuser/oaitt-gigaam 1.0.0

# 4. Запуск
docker-compose -f docker-compose.cpu.yml up -d
```

Подробнее:
- [`DOCKER_GIGAAM.md`](DOCKER_GIGAAM.md) — документация по Docker
- [`DOCKER_BUILD.md`](DOCKER_BUILD.md) — инструкции по сборке
- [`build.sh`](build.sh) — скрипт сборки с поддержкой multi-platform

---

## Архитектура

```
src/
├── __init__.py                     Версия пакета
├── app.py                          FastAPI приложение и жизненный цикл
├── config.py                       Конфигурация из переменных окружения
│
├── asr/                            ASR движки
│   ├── base.py                     Абстрактный базовый класс ASRModel
│   ├── transformers.py             Реализация на HuggingFace Transformers
│   ├── whisperx.py                 Реализация на WhisperX
│   ├── gigaam.py                   GigaAM native (PyTorch)
│   ├── gigaam_mlx.py               GigaAM v3 на MLX (Apple Silicon)
│   ├── gigaam_multilingual_mlx.py  GigaAM Multilingual на MLX
│   ├── pool.py                     Пул воркеров для параллельного инференса
│   └── factory.py                  Фабрика создания моделей
│
├── models/                         Pydantic модели
│   ├── schemas.py                  Основные модели (Segment, TranscriptionResponse)
│   └── openai.py                   OpenAI-совместимые модели
│
├── routes/                         HTTP маршруты
│   ├── health.py                   GET /health - проверка здоровья
│   ├── asr.py                      POST /asr - основной эндпоинт
│   └── openai.py                   POST /v1/audio/transcriptions
│
├── services/                       Бизнес-логика
│   ├── performance.py              Отслеживание производительности
│   ├── timeout.py                  Управление таймаутами
│   ├── memory_monitor.py           Мониторинг памяти
│   └── debug.py                    Отладочное логирование
│
└── utils/                          Утилиты
    ├── audio.py                    Загрузка и обработка аудио
    ├── chunking.py                 Разбиение длинного аудио
    ├── device.py                   Работа с устройствами (CUDA/MPS/CPU)
    └── formatters.py               Форматирование вывода (SRT/VTT/TSV)
```

### Основные компоненты

| Компонент | Описание |
|-----------|----------|
| `ASRModel` | Абстрактный базовый класс для ASR движков |
| `TransformersASR` | Реализация на HuggingFace Transformers |
| `GigaAMASR` | Реализация на GigaAM (опционально, требует установки пакета GigaAM) |
| `WhisperXASR` | Реализация на WhisperX с выравниванием слов |
| `PerformanceTracker` | Отслеживание производительности для адаптивных таймаутов |
| `TranscriptionResponse` | Единый формат ответа с сегментами и метриками |

---

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd oaitt # Open AI Transformer Transcriber

# Основное окружение: GigaAM Native / MLX / Transformers, Whisper Large V3.
# Создаёт venv, ставит зависимости, качает модели GigaAM.
./prepare-gigaam.sh

# Multilingual модели (~3GB, опционально)
GIGAAM_MULTILINGUAL=1 ./prepare.sh
```

**Два виртуальных окружения.** WhisperX пинит `torch~=2.8.0`, а GigaAM требует
`torch>=2.11.0` (фиксы утечек MPS на Apple Silicon) — в одном окружении они не уживаются:

| Скрипт | venv | Движки |
|--------|------|--------|
| `./prepare-gigaam.sh` | `venv` | GigaAM Native / MLX / Transformers, Whisper Large V3 |
| `./prepare-whisperx.sh` | `venv-whisperx` | WhisperX |

Скрипты `run_*.sh` сами выбирают нужный интерпретатор и подскажут, какой
prepare-скрипт запустить, если окружения нет.

### Бенчмарк

```bash
./benchmark.sh # все движки, полный файл, HTML-отчёт
./benchmark.sh --mode short -i 5 # 20s аудио, 5 итераций
./benchmark.sh -s run_gigaam_asr.sh # один движок
```

Отчёты пишутся в `bench_results/` (HTML + JSON), последний доступен как
`bench_results/latest.html`.

### GigaAM

GigaAM — это высококачественная модель распознавания русской речи от команды SberDevices. OAITT поддерживает два способа интеграции:

#### Способ 1: GigaAM Native (рекомендуется для максимальной скорости)

Использует оригинальный пакет `gigaam` напрямую. Это самый быстрый вариант (~327x realtime с CTC).

```bash
# Инициализация submodule
git submodule update --init --recursive

# Запуск с GigaAM Native
./run_gigaam_asr.sh

# Или вручную:
export PYTHONPATH="./vendor/gigaam:$PYTHONPATH"
ASR_ENGINE=gigaam GIGAAM_MODEL=v3_e2e_ctc python main.py
```

Доступные модели для native режима:
- `v3_e2e_rnnt` — лучшее качество с пунктуацией
- `v3_e2e_ctc` — end-to-end с пунктуацией (по умолчанию)
- `v3_rnnt`, `v3_ctc` — без пунктуации
- `v2_rnnt`, `v2_ctc` — предыдущая версия
- `multilingual_ctc` — 70+ языков, 220M энкодер, без пунктуации
- `multilingual_large_ctc` — 70+ языков, 600M энкодер, выше качество

##### GigaAM Multilingual

Претрейн на 2M часов, 70+ языков, charwise CTC декодер. Лучший WER на русском,
казахском, киргизском и узбекском; средний на английском. Пунктуацию не проставляет.

```bash
# Скачать multilingual модели (~3GB)
GIGAAM_MULTILINGUAL=1 ./prepare.sh

# Запуск (220M по умолчанию)
./run_gigaam_multilingual.sh

# Большая версия (600M)
GIGAAM_MODEL=multilingual_large_ctc ./run_gigaam_multilingual.sh
```

#### Способ 2: GigaAM MLX (Apple Silicon, без PyTorch)

Порты GigaAM на MLX — заметно быстрее PyTorch-версий тех же моделей на Apple Silicon.

**GigaAM v3** (`vendor/gigaam-mlx`, submodule):

```bash
./run_gigaam_mlx_asr.sh # RNNT (по умолчанию)
GIGAAM_MLX_MODEL_TYPE=ctc ./run_gigaam_mlx_asr.sh # CTC
```

**GigaAM Multilingual Large** (пакет `gigaam-multilingual-mlx` с PyPI, веса с HuggingFace):

```bash
./run_gigaam_multilingual_mlx.sh # int8 (по умолчанию)
GIGAAM_ML_MLX_VARIANT=fp16 ./run_gigaam_multilingual_mlx.sh
```

Это тот же 600M `multilingual_large_ctc`, но через MLX — вдвое быстрее PyTorch-версии.
Варианты `int8` и `fp16` — квантизация одних и тех же весов, скорость одинаковая,
`int8` занимает вдвое меньше памяти. Малой 220M версии в MLX нет.

#### Способ 3: GigaAM через Transformers (удобнее для установки)

Загружает модель через Hugging Face, не требует submodule:

```bash
# Запуск GigaAM через transformers
./run_gigaam.sh

# Или вручную:
ASR_ENGINE=transformers WHISPER_MODEL=ai-sage/GigaAM-v3 GIGAAM_REVISION=e2e_ctc python main.py
```

> **Сейчас не работает с transformers 5.x**: кастомный код модели `ai-sage/GigaAM-v3`
> вызывает `.item()` на meta-тензоре при инициализации (`Tensor.item() cannot be called on
> meta tensors`). Чинится на стороне модели. Используйте native — та же модель, быстрее.
>
> Для этого способа также требуются: `hydra-core`, `omegaconf`, `torchaudio`, `pyannote.audio`.

#### Полезные ссылки

- GitHub: https://github.com/salute-developers/GigaAM
- Hugging Face: https://huggingface.co/ai-sage/GigaAM-v3
- Colab: https://github.com/salute-developers/GigaAM/blob/main/colab_example.ipynb

---

## Конфигурация

Все настройки задаются через переменные окружения:

### Основные настройки

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `ASR_ENGINE` | `transformers` | ASR движок: `transformers`, `whisperx` или `gigaam` |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Модель для Transformers |
| `WHISPERX_MODEL` | `large-v3` | Модель для WhisperX |
| `GIGAAM_MODEL` | `v3_e2e_ctc` | Модель для GigaAM Native |
| `GIGAAM_REVISION` | `e2e_rnnt` | Ревизия для GigaAM через Transformers |
| `GIGAAM_MAX_SHORT_AUDIO_SEC` | `25.0` | Порог (сек) для chunked транскрипции |
| `GIGAAM_CHUNK_SEC` | `30` | Размер чанка для длинных аудио (сек) |
| `DEFAULT_LANGUAGE` | `ru` | Язык по умолчанию (если не указан в API) |
| `DEVICE` | `auto` | Устройство: `auto`, `cuda`, `cpu`, `mps` |
| `COMPUTE_TYPE` | `float32` | Тип вычислений для WhisperX |

### Сервер

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `HOST` | `0.0.0.0` | Хост для привязки сервера |
| `PORT` | `9007` | Порт для привязки сервера |
| `MODEL_CACHE_DIR` | `./data` | Директория для кэша моделей |
| `DEBUG_LOG_DIR` | *(отключено)* | Директория для отладочных логов |
| `MODEL_IDLE_TIMEOUT` | `0` | Таймаут выгрузки модели (0 = никогда) |

### Адаптивные таймауты

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `TIMEOUT_ENABLED` | `true` | Включить адаптивные таймауты |
| `TIMEOUT_MULTIPLIER` | `2.0` | Множитель ожидаемого времени |
| `TIMEOUT_MIN_SECONDS` | `30.0` | Минимальный таймаут |
| `TIMEOUT_MAX_SECONDS` | `300.0` | Максимальный таймаут |
| `TIMEOUT_HISTORY_SIZE` | `100` | Размер истории для расчёта |

### Фильтрация по уверенности

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `CONFIDENCE_FILTER_ENABLED` | `false` | Фильтровать низкокачественные результаты |
| `CONFIDENCE_AVG_LOGPROB_THRESHOLD` | `-1.0` | Порог avg_logprob |
| `CONFIDENCE_NO_SPEECH_THRESHOLD` | `0.6` | Порог no_speech_prob |
| `CONFIDENCE_WORD_SCORE_THRESHOLD` | `0.5` | Порог оценки слов |
| `CONFIDENCE_WORD_PROB_THRESHOLD` | `0.4` | Порог вероятности слов |

### Проверка скорости символов (chars/sec)

Для обнаружения случаев, когда модель выдаёт чрезмерно много текста относительно длительности аудио, введены дополнительные параметры конфигурации.

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `MAX_CHARS_PER_SECOND` | `25.0` | Ожидаемая базовая скорость символов (символов в секунду) |
| `CHARS_PER_SECOND_MULTIPLIER` | `3.0` | Множитель порога; если наблюдаемая скорость > base * multiplier — помечаем результат как подозрительный |
| `CHARS_PER_SECOND_MIN_AUDIO_SEC` | `0.5` | Минимальная длительность аудио (с) для применения проверки |

Если наблюдаемая скорость символов (len(text) / audio_duration_sec) превышает `MAX_CHARS_PER_SECOND * CHARS_PER_SECOND_MULTIPLIER` и длительность аудио не меньше `CHARS_PER_SECOND_MIN_AUDIO_SEC`, то транскрипция помечается как потенциально ошибочная — в поле `confidence.high_char_rate` ставится `true`, в `confidence.rejection_reasons` добавляется причина, а `confidence.is_reliable` устанавливается в `false`. При включённой опции `CONFIDENCE_FILTER_ENABLED` сервис может вернуть пустой результат для таких транскрипций (см. `/asr` и `/v1/audio/transcriptions`).

---

## Запуск

### Базовый запуск

```bash
# Запуск с Whisper Large V3 (Transformers)
./run_whisper_large_v3.sh

# Запуск с GigaAM Native (самый быстрый для русского)
./run_gigaam_asr.sh

# Запуск с GigaAM через Transformers
./run_gigaam.sh

# Запуск с WhisperX
./run_whisperx_large_v3.sh

# Или напрямую:
python main.py # По умолчанию Whisper
ASR_ENGINE=whisperx python main.py # WhisperX
ASR_ENGINE=gigaam python main.py # GigaAM Native

# На Apple Silicon
DEVICE=mps python main.py

# С отладочным логированием
DEBUG_LOG_DIR=./debug_logs python main.py
```

### Batch-обработка

```bash
# Batch-транскрипция с GigaAM Native
./run_gigaam_asr_batch.sh

# Batch-транскрипция с GigaAM через Transformers
./run_gigaam_batch.sh

# С указанием директорий:
SAMPLES_DIR=my_audio OUTPUT_DIR=results ./run_gigaam_asr_batch.sh
```

### Использование как модуля

```python
from src.app import app, run_server

# Запуск сервера
run_server(host="0.0.0.0", port=9007)
```

---

## API Endpoints

### `GET /health`

Проверка здоровья сервиса.

```bash
curl http://localhost:9007/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "engine": "whisperx",
  "timeout_enabled": true,
  "performance": {
    "samples": 42,
    "avg_ratio": 0.0853,
    "avg_speed": 11.72
  }
}
```

### `POST /asr`

Основной эндпоинт транскрипции.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `audio_file` | file | *required* | Аудиофайл |
| `output` | string | `json` | Формат: `text`, `json`, `vtt`, `srt`, `tsv` |
| `task` | string | `transcribe` | Задача: `transcribe`, `translate` |
| `language` | string | *auto* | Код языка |
| `word_timestamps` | bool | `true` | Временные метки слов |

**Пример:**
```bash
curl -X POST "http://localhost:9007/asr" \
  -F "audio_file=@audio.wav" \
  -F "output=json"
```

### `POST /v1/audio/transcriptions`

**OpenAI-совместимый эндпоинт** — drop-in замена для OpenAI Whisper API.

**Параметры:**

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `file` | file | *required* | Аудиофайл |
| `model` | string | `whisper-1` | Модель (игнорируется) |
| `language` | string | *auto* | Код языка |
| `response_format` | string | `json` | Формат: `json`, `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities[]` | string[] | — | Гранулярность: `word`, `segment` |

**Пример:**
```bash
curl -X POST "http://localhost:9007/v1/audio/transcriptions" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word"
```

**Ответ (verbose_json):**
```json
{
  "text": "Hello world",
  "task": "transcribe",
  "language": "en",
  "duration": 2.5,
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.5, "prob": 0.95},
    {"word": "world", "start": 0.6, "end": 1.0, "prob": 0.92}
  ],
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.0,
      "text": "Hello world",
      "avg_logprob": -0.3,
      "no_speech_prob": 0.01
    }
  ]
}
```

---

## Метрики уверенности

Сервис предоставляет метрики для оценки качества транскрипции:

| Метрика | Описание | Хорошие значения |
|---------|----------|------------------|
| `avg_logprob` | Средняя log-вероятность токенов | > -0.5 |
| `chars_per_second` | Наблюдаемая скорость символов (len(text) / duration) | Зависит от языка и темпа речи (обычно < 25) |
| `chars_per_second_ratio` | Отношение к базовой скорости (`chars_per_second / MAX_CHARS_PER_SECOND`) | < multiplier (например, < 3.0) |
| `high_char_rate` | Флаг, если скорость символов аномально велика | `false` (ожидается) |

Дополнительно в модели `Segment` добавлено поле `chars_per_second` — скорость символов внутри сегмента (len(segment.text) / (segment.end - segment.start)). Эти поля помогают выявлять ситуации, когда модель генерирует очень много текста за короткое время (возможная галлюцинация или ошибочное декодирование).
| `no_speech_prob` | Вероятность отсутствия речи | < 0.3 |
| `avg_word_score` | Средняя оценка выравнивания слов | > 0.7 |
| `avg_word_prob` | Средняя вероятность слов | > 0.6 |
| `low_prob_word_ratio` | Доля слов с низкой вероятностью | < 0.3 |

### Интерпретация `prob` (WhisperX)

| Вероятность | Интерпретация |
|-------------|---------------|
| > 0.8 | Высокая уверенность |
| 0.5 - 0.8 | Средняя уверенность |
| 0.2 - 0.5 | Низкая уверенность |
| < 0.2 | Слово вероятно ошибочно |

---

## Адаптивные таймауты

Сервис отслеживает производительность и автоматически прерывает запросы, которые занимают слишком много времени:

1. Записывается соотношение `processing_time / audio_duration` для каждой успешной транскрипции
2. Для новых запросов вычисляется ожидаемое время на основе истории
3. Таймаут = `expected_time * TIMEOUT_MULTIPLIER`
4. При превышении возвращается HTTP 408

**Это помогает обнаруживать:**
- Галлюцинации модели (бесконечные циклы)
- Зависания из-за повреждённого аудио
- Аномально медленную обработку

---

## Apple Silicon (MPS)

| Движок | Поддержка ускорения |
|--------|---------------------|
| `gigaam_mlx` | MLX (Metal) — нативный бэкенд Apple Silicon, без PyTorch |
| `gigaam_multilingual_mlx` | MLX (Metal) — нативный бэкенд Apple Silicon, без PyTorch |
| `gigaam` | MPS (float32) |
| `transformers` | MPS (float16) |
| `whisperx` | Fallback на CPU (ctranslate2 не поддерживает MPS) |

Для максимальной скорости на Apple Silicon используйте MLX-движки:

```bash
./run_gigaam_mlx_asr.sh                 # GigaAM v3
./run_gigaam_multilingual_mlx.sh        # GigaAM Multilingual
```

---

## Отладка

При установке `DEBUG_LOG_DIR` сервис сохраняет каждый запрос:

```bash
DEBUG_LOG_DIR=./debug_logs python run.py
```

Создаются файлы:
- `{timestamp}_{filename}.wav` — аудио (16kHz mono)
- `{timestamp}_{filename}.json` — результат транскрипции

---

## Зависимости

Основные зависимости:
- **FastAPI** — веб-фреймворк
- **PyTorch** — глубокое обучение
- **Transformers** — модели Whisper
- **WhisperX** — выравнивание слов
- **librosa** — обработка аудио

Обновление зависимостей:
```bash
pip install pip-tools
pip-compile --upgrade requirements.in -o requirements.txt
```

---

## Лицензия

MIT License

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)

---

## Благодарности

### GigaAM Team

Особая благодарность команде **SberDevices / Salute** за создание и открытый доступ к модели **GigaAM** — высококачественной модели распознавания русской речи:

- **Организация**: [SberDevices](https://sberdevices.ru/)
- **Репозиторий**: [github.com/salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
- **Hugging Face**: [huggingface.co/ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
- **Лицензия**: MIT

GigaAM обеспечивает:
- Высокое качество распознавания русской речи
- Поддержку пунктуации (модели e2e)
- Высокую скорость работы (~327x realtime)
- Поддержку длинных аудио через VAD-сегментацию

### OpenAI Whisper

Благодарность команде **OpenAI** за модели Whisper, которые послужили основой для OpenAI-совместимого API.

### Hugging Face

Благодарность **Hugging Face** за библиотеку Transformers и инфраструктуру для распространения моделей.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  <b>OAITT</b> — Open AI Transformer Transcriber<br>
  Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)<br>
  Made with for the speech recognition community
</p>
