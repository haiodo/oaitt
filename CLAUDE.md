# OAITT

Сервис распознавания речи: GigaAM на MLX, OpenAI-совместимый API. Две реализации -
Python (`src/`, в проде) и нативная на Swift (`swift/`). Документация - `docs/INDEX.md`,
план работ - `docs/roadmap.md`.

## После любых правок

```bash
make check      # формат, линтер, сборка Swift, тесты Python
```

Отдельно:

```bash
make format     # swift format -i
make lint       # swift format lint + swiftlint --strict
make test       # pytest
make swift-build
```

`make lint` падает на ошибках swiftlint, но не на предупреждениях. Предупреждения о
`baseAddress!` внутри `withUnsafeBufferPointer` осознанные - там разыменование безопасно.

Правки в Swift обязательно прогонять через `make format` до коммита: конфиг
`swift/.swift-format` задаёт длину строки 100 и порядок импортов, иначе диффы шумят.

## Что где

| | |
|---|---|
| `src/` | Python-сервис: движки в `src/asr/`, роуты в `src/routes/` |
| `swift/Sources/GigaAM/` | Модель, аудио, mel, токенизатор, кеш, телеметрия |
| `swift/Sources/oaitt-swift/` | CLI: `transcribe`, `serve`, `bench`, `balance` |
| `swift/Sources/OAITT/` | Приложение для строки меню |
| `vendor/gigaam-mlx` | Сабмодуль, менять нельзя - изменения только в `src/` |

## Замеры

Числа в `docs/benchmarks.md` сняты на M4 Max. Перед тем как менять что-то ради скорости,
сверьтесь с ними: часть очевидных идей там уже проверена и отвергнута с цифрами (например
общая очередь чанков - она убивает перекрытие CPU и GPU).

Мерить на одном и том же файле без `--cache-size 0` бессмысленно - попадания в кеш дают
абсурдные цифры вроде 44000x realtime.

## Модели

Веса в `data/gigaam_mlx/<type>/`, в репозиторий не входят: `make prepare`. Приложение
качает их само с HuggingFace.
