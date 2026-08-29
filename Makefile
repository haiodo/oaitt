# OAITT - точка входа для всех операций. Скрипты остаются на месте, здесь их оболочка.
#
#   make            список целей
#   make prepare    зависимости и веса
#   make run        Python-сервис (порт 9007)
#   make app        приложение для macOS

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON      ?= venv/bin/python
PYTEST      ?= venv/bin/python -m pytest
SWIFT_DIR   := swift
SWIFT_BIN   := $(SWIFT_DIR)/.build/release/oaitt-swift
APP         := $(SWIFT_DIR)/.build/OAITT.app
MODEL_DIR   ?= data/gigaam_mlx
MODEL_TYPE  ?= rnnt
PORT        ?= 9007
SWIFT_PORT  ?= 9007
FILE        ?=

.PHONY: help
help: ## Показать эти цели
	@awk 'BEGIN {FS = ":.*## "; section=""} \
		/^## / {printf "\n\033[1m%s\033[0m\n", substr($$0, 4); next} \
		/^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nПеременные: MODEL_TYPE=%s PORT=%s SWIFT_PORT=%s FILE=<путь>\n\n" "$(MODEL_TYPE)" "$(PORT)" "$(SWIFT_PORT)"

## Подготовка

.PHONY: prepare
prepare: ## Окружение Python и веса GigaAM MLX
	./prepare-gigaam.sh

.PHONY: prepare-all
prepare-all: ## Все движки, включая PyTorch-версии
	./prepare.sh

.PHONY: prepare-whisperx
prepare-whisperx: ## Отдельное окружение для WhisperX
	./prepare-whisperx.sh

## Python-сервис

.PHONY: run
run: ## GigaAM MLX на порту $(PORT)
	PORT=$(PORT) GIGAAM_MLX_MODEL_TYPE=$(MODEL_TYPE) ./run_gigaam_mlx_asr.sh

.PHONY: run-workers
run-workers: ## То же, но пулом процессов-воркеров
	./run_gigaam_mlx_with_workers.sh

.PHONY: run-native
run-native: ## GigaAM на PyTorch, без MLX
	./run_gigaam_asr.sh

.PHONY: run-multilingual
run-multilingual: ## Multilingual MLX: русский, казахский, киргизский, узбекский
	./run_gigaam_multilingual_mlx.sh

.PHONY: run-whisper
run-whisper: ## Whisper Large V3 через transformers
	./run_whisper_large_v3.sh

## Swift

.PHONY: swift-build
swift-build: ## Собрать CLI и metallib
	swift build -c release --package-path $(SWIFT_DIR) --product oaitt-swift
	$(SWIFT_DIR)/build-metallib.sh

.PHONY: metallib
metallib: ## Собрать только Metal-шейдеры из исходников mlx-swift
	$(SWIFT_DIR)/build-metallib.sh

.PHONY: swift-run
swift-run: ## Swift-сервис на порту $(SWIFT_PORT)
	PORT=$(SWIFT_PORT) GIGAAM_MLX_MODEL_TYPE=$(MODEL_TYPE) ./run_swift_asr.sh

.PHONY: swift-transcribe
swift-transcribe: swift-build ## Расшифровать файл: make swift-transcribe FILE=audio.ogg
	@test -n "$(FILE)" || { echo "Укажите FILE=<путь к аудио>"; exit 1; }
	$(SWIFT_BIN) transcribe "$(FILE)" --model-cache-dir $(MODEL_DIR) --model-type $(MODEL_TYPE) --segments

.PHONY: swift-bench
swift-bench: swift-build ## Замер пропускной способности: make swift-bench FILE=audio.wav
	@test -n "$(FILE)" || { echo "Укажите FILE=<путь к аудио>"; exit 1; }
	$(SWIFT_BIN) bench "$(FILE)" --model-cache-dir $(MODEL_DIR) --model-type $(MODEL_TYPE) --concurrency 4 --iterations 16

.PHONY: app
app: ## Собрать OAITT.app
	$(SWIFT_DIR)/build-app.sh

.PHONY: app-run
app-run: app ## Собрать и запустить приложение
	open $(APP)

## Проверки

.PHONY: test
test: ## Тесты Python
	PYTHONPATH=vendor/gigaam-mlx MODEL_CACHE_DIR=$(PWD)/data $(PYTEST) tests/ -q

.PHONY: bench
bench: ## Полный бенчмарк всех движков с HTML-отчётом
	./benchmark.sh

.PHONY: format
format: ## Отформатировать Swift (swift format из toolchain)
	swift format -i -r $(SWIFT_DIR)/Sources
	@echo "Отформатировано"

.PHONY: lint
lint: ## Проверить Swift: формат и swiftlint
	@swift format lint -r $(SWIFT_DIR)/Sources && echo "swift format: чисто"
	@command -v swiftlint >/dev/null || { echo "swiftlint не установлен: brew install swiftlint"; exit 1; }
	@cd $(SWIFT_DIR) && swiftlint lint --quiet --strict || \
		{ echo; echo "swiftlint нашёл ошибки"; exit 1; }
	@echo "swiftlint: чисто"

.PHONY: check
check: format lint swift-build test ## Формат, линт, сборка, тесты - прогонять после правок

## Docker

.PHONY: docker-build
docker-build: ## Собрать образ: make docker-build IMAGE=user/oaitt TAG=1.0.0
	./build.sh $(if $(IMAGE),$(IMAGE),myuser/oaitt-gigaam) $(if $(TAG),$(TAG),latest)

.PHONY: docker-run
docker-run: ## Запустить контейнер (CPU)
	./docker-run-cpu.sh

.PHONY: docker-run-mlx
docker-run-mlx: ## Запустить контейнер (MLX)
	./docker-run-mlx-cpu.sh

.PHONY: docker-test
docker-test: ## Проверить собранный образ
	./test-docker.sh

## Уборка

.PHONY: clean
clean: ## Удалить артефакты сборки Swift
	rm -rf $(SWIFT_DIR)/.build

.PHONY: clean-all
clean-all: clean ## Плюс venv и кеши Python
	rm -rf venv venv-whisperx __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
