# OAITT with GigaAM в Docker (CPU режим)

Запуск OAITT с GigaAM моделями в Docker контейнере без GPU.

## Подготовка

### 1. Инициализация и скачивание моделей

```bash
# Запустить подготовку (скачивает ~900MB моделей)
./prepare.sh
```

Этот скрипт:
- Инициализирует git submodule `vendor/gigaam`
- Скачивает модели GigaAM в `data/gigaam/`:
  - `v3_e2e_rnnt.ckpt` (~450MB) - лучшее качество с пунктуацией
  - `v3_e2e_ctc.ckpt` (~440MB) - CTC версия

> **Примечание:** Если модели уже скачаны, скрипт пропустит загрузку.

### 2. Сборка Docker образа

```bash
# Стандартная сборка (может занять 5-10 минут)
docker build -f Dockerfile.cpu -t oaitt-gigaam:cpu .

# Или с BuildKit для ускорения повторных сборок
DOCKER_BUILDKIT=1 docker build -f Dockerfile.cpu -t oaitt-gigaam:cpu .
```

### 3. Запуск

**Через docker-compose (рекомендуется):**
```bash
docker-compose -f docker-compose.cpu.yml up -d
```

**Через скрипт:**
```bash
./docker-run-cpu.sh
```

**Вручную:**
```bash
docker run -d \
    --name oaitt-gigaam-cpu \
    -p 9007:9007 \
    -e DEVICE=cpu \
    -e ASR_ENGINE=gigaam \
    -e GIGAAM_MODEL=v3_e2e_rnnt \
    -v $(pwd)/data:/app/data \
    oaitt-gigaam:cpu
```

## Проверка работы

```bash
# Health check
curl http://localhost:9007/health

# Тестовая транскрипция
curl -X POST http://localhost:9007/v1/audio/transcriptions \
    -H "Content-Type: multipart/form-data" \
    -F "file=@samples/test.wav" \
    -F "model=gigaam"
```

## Доступные модели

| Модель | Описание | Размер |
|--------|----------|--------|
| `v3_e2e_rnnt` | Лучшее качество, пунктуация | ~450MB |
| `v3_e2e_ctc` | CTC версия, пунктуация | ~440MB |
| `v3_rnnt` | Без пунктуации | ~420MB |
| `v3_ctc` | CTC без пунктуации | ~410MB |

Укажите модель через переменную `GIGAAM_MODEL`.

## Структура образа

```
oaitt-gigaam:cpu
├── /app/main.py              # Точка входа
├── /app/src/                 # Исходный код
├── /app/vendor/gigaam/       # Пакет gigaam (установлен)
├── /app/data/gigaam/         # Модели (встроены в образ)
│   ├── v3_e2e_rnnt.ckpt
│   └── v3_e2e_ctc.ckpt
└── Python 3.11 + PyTorch CPU
```

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `DEVICE` | Устройство | `cpu` |
| `ASR_ENGINE` | Движок ASR | `gigaam` |
| `GIGAAM_MODEL` | Модель GigaAM | `v3_e2e_rnnt` |
| `MODEL_CACHE_DIR` | Кэш моделей | `/app/data` |
| `MODEL_WORKERS` | Воркеры | `1` |
| `PORT` | Порт | `9007` |
| `AUTH_TOKEN` | Токен авторизации | `key` |

## Управление

```bash
# Логи
docker logs -f oaitt-gigaam-cpu

# Остановка
docker stop oaitt-gigaam-cpu

# Перезапуск
docker restart oaitt-gigaam-cpu

# Удаление контейнера
docker rm -f oaitt-gigaam-cpu
```

## Примечания

- **Первый старт:** модель загружается из `/app/data/gigaam/` (встроена в образ)
- **CPU режим:** медленнее GPU, но работает на любом железе
- **Расход памяти:** ~2-4GB RAM
- **Размер образа:** ~2.5GB (включая модели)
