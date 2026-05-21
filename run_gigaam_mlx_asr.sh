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
export GIGAAM_MLX_MODEL_TYPE=${GIGAAM_MLX_MODEL_TYPE:-ctc}

# Add vendor/gigaam-mlx to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam-mlx:${PYTHONPATH}"

# Auto-detect locally converted weights (data/gigaam_mlx/{ctc,rnnt}/)
# Convert via: python scripts/convert_gigaam_to_mlx.py --model {ctc,rnnt,both}
LOCAL_MLX_DIR="${SCRIPT_DIR}/data/gigaam_mlx/${GIGAAM_MLX_MODEL_TYPE}"
if [ -z "${GIGAAM_MLX_REPO_ID}" ] && [ -f "${LOCAL_MLX_DIR}/weights.safetensors" ]; then
    export GIGAAM_MLX_REPO_ID="${LOCAL_MLX_DIR}"
    echo "Using locally converted MLX weights: ${LOCAL_MLX_DIR}"
fi

python main.py
