#!/bin/bash
#
# Run OAITT with GigaAM-Multilingual on MLX (Apple Silicon).
#
# GigaAM-Multilingual: 600M encoder, charwise CTC, pre-trained on 2M hours
# across 70+ languages. Best-in-class WER on Russian, Kazakh, Kyrgyz and Uzbek.
# Does not emit punctuation.
#
# Weights come from HuggingFace (ai-babai/gigaam-multilingual-mlx) via the
# gigaam-multilingual-mlx package - no submodule, no PyTorch.
#
# Env vars:
#   GIGAAM_ML_MLX_VARIANT=int8|fp16   - weights variant (default: int8).
#                                       Same speed, int8 uses ~half the memory.
#   GIGAAM_ML_MLX_CHUNK_SEC=20.0      - chunk length in seconds
#   GIGAAM_ML_MLX_OVERLAP_SEC=2.0     - overlap between chunks
#   GIGAAM_ML_MLX_MODEL_DIR=<path>    - local weights instead of HuggingFace
#   GIGAAM_ML_MLX_LOCK_FREE=true      - in-process lock-free threading
#   MODEL_WORKERS=N                   - parallel model instances (default: 1)
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh gigaam_multilingual_mlx

export ASR_ENGINE=gigaam_multilingual_mlx
export GIGAAM_ML_MLX_VARIANT=${GIGAAM_ML_MLX_VARIANT:-int8}

# Кэш HuggingFace держим в проекте, а не в ~/.cache - веса 1.1GB.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data}"

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
