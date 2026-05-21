#!/bin/bash
#
# Run OAITT with native GigaAM-MLX ASR engine (Apple Silicon).
# Uses gigaam_mlx package from vendor/gigaam-mlx submodule.
#
# Requirements:
#   - macOS with Apple Silicon (M1/M2/M3/M4)
#   - ffmpeg installed (brew install ffmpeg)
#   - pip install mlx librosa huggingface_hub sentencepiece numpy
#
# Available model types:
#   - ctc  (default, ~330x realtime, good quality)
#   - rnnt (~77x realtime, higher quality)
#
# Env vars:
#   GIGAAM_MLX_MODEL_TYPE=ctc|rnnt   - model variant (default: ctc)
#   GIGAAM_MLX_REPO_ID=<repo>        - override HF repo (optional)
#   GIGAAM_MLX_CHUNK_SEC=20.0        - max chunk size in seconds
#   GIGAAM_MLX_LOCK_FREE=true|false  - in-process lock-free threading (default: true).
#                                      MLX операции lazy + thread-safe на построение
#                                      графа, GPU runtime сериализует исполнение.
#                                      Даёт линейный throughput по клиентам без
#                                      MODEL_WORKERS. Set to false при нехватке
#                                      unified memory (каждый клиент держит свои
#                                      activations).
#   MODEL_WORKERS=N                  - parallel model instances (default: 1)
#

export ASR_ENGINE=gigaam_mlx
export GIGAAM_MLX_MODEL_TYPE=${GIGAAM_MLX_MODEL_TYPE:-rnnt}

# Add vendor/gigaam-mlx to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam-mlx:${PYTHONPATH}"

# Pin model cache to project's data/ - иначе ./data резолвится от cwd запуска
# и engine может не найти локальные веса -> начнёт качать с HuggingFace.
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${SCRIPT_DIR}/data}"

# Engine сам зарезолвит локальные MLX веса в ${MODEL_CACHE_DIR}/gigaam_mlx/<type>/.
# Если их нет, но есть PyTorch checkpoint в ${MODEL_CACHE_DIR}/gigaam/v3_e2e_<type>.ckpt
# - сконвертирует автоматически. Иначе скачает с HuggingFace.
# Принудительно задать путь: GIGAAM_MLX_REPO_ID=/path/to/weights

# cd в project root - main.py и scripts.convert_gigaam_to_mlx требуют корректного cwd
cd "${SCRIPT_DIR}"
exec python main.py
