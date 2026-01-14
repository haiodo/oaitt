# 🎙️ OAITT — Open AI Transformer Transcriber

**Высокопроизводительный сервис распознавания речи с поддержкой Whisper и GigaAM**

OAITT — это speech-to-text сервис транскрипции с поддержкой нескольких ASR движков, OpenAI-совместимым API и расширенными метриками качества.

---

## ✨ Возможности

- 🚀 **Четыре ASR движка**: Hugging Face Transformers, WhisperX, GigaAM (via HF) и GigaAM Native
- 🔌 **OpenAI-совместимый API**: Drop-in замена для OpenAI Whisper API
- ⏱️ **Точные временные метки**: На уровне слов (с WhisperX)
- 📊 **Метрики уверенности**: Оценка качества транскрипции
- ⏰ **Адаптивные таймауты**: Защита от зависания модели
- 🍎 **Apple Silicon**: Поддержка MPS для Mac
- 📝 **Множество форматов**: JSON, текст, SRT, VTT, TSV
- ⚡ **Высокая производительность**: До 150x realtime с GigaAM Native на Apple Silicon

---

## 📈 Производительность

### Тестовая конфигурация

| Параметр | Значение |
|----------|----------|
| **Модель** | MacBook Pro (Mac16,5) |
| **Чип** | Apple M4 Max |
| **Ядра** | 16 (12 производительных + 4 энергоэффективных) |
| **Память** | 48 GB |

### Результаты бенчмарка

Тестовое аудио: **137.4 секунд**, режим: full, 5 итераций:

| Движок | Avg (s) | Min (s) | Max (s) | Скорость | Статус |
|--------|---------|---------|---------|----------|--------|
| **GigaAM (Native)** | **0.52** | 0.44 | 0.65 | **262.37x** | ✓ OK |
| GigaAM (Transformers) | 1.15 | 1.09 | 1.39 | 119.08x | ✓ OK |
| Whisper Large V3 (Transformers) | 13.78 | 13.47 | 14.88 | 9.97x | ✓ OK |
| WhisperX Large V3 | 35.75 | 34.73 | 39.15 | 3.84x | ✓ OK |

> 💡 **GigaAM Native — самый быстрый вариант** для русского языка (262x realtime), использует прямую передачу тензоров без временных файлов.

---

## 🏗️ Архитектура

```
src/
├── __init__.py              # Версия пакета
├── app.py                   # FastAPI приложение и жизненный цикл
├── config.py                # Конфигурация из переменных окружения
│
├── asr/                     # ASR движки
│   ├── base.py              # Абстрактный базовый класс ASRModel
│   ├── transformers.py      # Реализация на HuggingFace Transformers
│   ├── whisperx.py          # Реализация на WhisperX
│   ├── gigaam.py            # Реализация на GigaAM (опционально)
│   └── factory.py           # Фабрика создания моделей
│
├── models/                  # Pydantic модели
│   ├── schemas.py           # Основные модели (Segment, TranscriptionResponse)
│   └── openai.py            # OpenAI-совместимые модели
│
├── routes/                  # HTTP маршруты
│   ├── health.py            # GET /health - проверка здоровья
│   ├── asr.py               # POST /asr - основной эндпоинт
│   └── openai.py            # POST /v1/audio/transcriptions
│
├── services/                # Бизнес-логика
│   ├── performance.py       # Отслеживание производительности
│   ├── timeout.py           # Управление таймаутами
│   └── debug.py             # Отладочное логирование
│
└── utils/                   # Утилиты
    ├── audio.py             # Загрузка и обработка аудио
    ├── device.py            # Работа с устройствами (CUDA/MPS/CPU)
    └── formatters.py        # Форматирование вывода (SRT/VTT/TSV)
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

## 🔧 Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd oaitt  # Open AI Transformer Transcriber

# Создание виртуального окружения
python3.13 -m venv venv
source ./venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### GigaAM

GigaAM — это высококачественная модель распознавания русской речи от команды SberDevices. OAITT поддерживает два способа интеграции:

#### Способ 1: GigaAM Native (рекомендуется для максимальной скорости)

Использует оригинальный пакет `gigaam` напрямую. Это самый быстрый вариант (~150x realtime).

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

#### Способ 2: GigaAM через Transformers (удобнее для установки)

Загружает модель через Hugging Face, не требует submodule:

```bash
# Запуск GigaAM через transformers
./run_gigaam.sh

# Или вручную:
ASR_ENGINE=transformers WHISPER_MODEL=ai-sage/GigaAM-v3 GIGAAM_REVISION=e2e_ctc python main.py
```

> ⚠️ Для этого способа могут потребоваться дополнительные зависимости: `hydra-core`, `omegaconf`, `torchaudio`.

#### Полезные ссылки

- 📦 GitHub: https://github.com/salute-developers/GigaAM
- 🤗 Hugging Face: https://huggingface.co/ai-sage/GigaAM-v3
- 📓 Colab: https://github.com/salute-developers/GigaAM/blob/main/colab_example.ipynb

---

## ⚙️ Конфигурация

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

## 🚀 Запуск

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
python main.py                           # По умолчанию Whisper
ASR_ENGINE=whisperx python main.py       # WhisperX
ASR_ENGINE=gigaam python main.py         # GigaAM Native

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

## 📡 API Endpoints

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

## 📊 Метрики уверенности

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

## ⏰ Адаптивные таймауты

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

## 🍎 Apple Silicon (MPS)

| Движок | Поддержка MPS |
|--------|---------------|
| `transformers` | ✅ Полная поддержка (float16) |
| `whisperx` | ⚠️ Fallback на CPU (ctranslate2 не поддерживает MPS) |

Для лучшей производительности на Apple Silicon используйте `transformers`:

```bash
ASR_ENGINE=transformers DEVICE=mps python run.py
```

---

## 🐛 Отладка

При установке `DEBUG_LOG_DIR` сервис сохраняет каждый запрос:

```bash
DEBUG_LOG_DIR=./debug_logs python run.py
```

Создаются файлы:
- `{timestamp}_{filename}.wav` — аудио (16kHz mono)
- `{timestamp}_{filename}.json` — результат транскрипции

---

## 📦 Зависимости

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

## 📄 Лицензия

MIT License

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)

---

## 🙏 Благодарности

### GigaAM Team

Особая благодарность команде **SberDevices / Salute** за создание и открытый доступ к модели **GigaAM** — высококачественной модели распознавания русской речи:

- 🏢 **Организация**: [SberDevices](https://sberdevices.ru/)
- 📦 **Репозиторий**: [github.com/salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
- 🤗 **Hugging Face**: [huggingface.co/ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
- 📄 **Лицензия**: MIT

GigaAM обеспечивает:
- Высокое качество распознавания русской речи
- Поддержку пунктуации (модели e2e)
- Высокую скорость работы (~150x realtime)
- Поддержку длинных аудио через VAD-сегментацию

### OpenAI Whisper

Благодарность команде **OpenAI** за модели Whisper, которые послужили основой для OpenAI-совместимого API.

### Hugging Face

Благодарность **Hugging Face** за библиотеку Transformers и инфраструктуру для распространения моделей.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  <b>OAITT</b> — Open AI Transformer Transcriber<br>
  Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)<br>
  Made with ❤️ for the speech recognition community
</p>
