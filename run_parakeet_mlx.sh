#!/bin/bash
#
# Run OAITT with NVIDIA Parakeet-TDT-v3 on MLX (Apple Silicon).
#
# Parakeet TDT v3: 600M FastConformer-TDT, 25 European languages with automatic
# language detection. WER 3.99% on the Golos common subset - better than GigaAM
# (6.69%). Word-level timestamps and per-token confidence are built in.
#
# Weights come from HuggingFace (mlx-community/parakeet-tdt-0.6b-v3) via the
# parakeet-mlx package - no PyTorch.
#
# Env vars:
#   PARAKEET_VARIANT=fp16|int8     - weights variant (default: fp16).
#                                    int8 = sonic-speech quantized encoder, +30% speed.
#   PARAKEET_CHUNK_SEC=120.0       - window length for long files
#   PARAKEET_OVERLAP_SEC=15.0      - overlap between windows
#   PARAKEET_MLX_LOCK_FREE=true    - in-process lock-free threading
#   PARAKEET_REPO_ID=<repo>        - custom HF repo or local weights dir
#   MODEL_WORKERS=N                - parallel model instances (default: 1)
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh parakeet_mlx

export ASR_ENGINE=parakeet_mlx
export PARAKEET_VARIANT=${PARAKEET_VARIANT:-fp16}

# Кэш HuggingFace держим в проекте, а не в ~/.cache - веса 1.2GB.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data}"

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
