#!/bin/bash
#
# Run OAITT with onnx-asr (ONNX Runtime, CPU) engine.
#
# GigaAM/Parakeet weights exported to ONNX by istupakov, run through onnxruntime -
# no PyTorch, no MLX. Works on any platform (x86_64/arm64, Linux/macOS/Windows).
#
# Env vars:
#   ONNX_ASR_MODEL=gigaam-v3-e2e-rnnt   - model name from the onnx-asr registry
#   ONNX_ASR_QUANTIZATION=int8|fp32     - "fp32" disables quantization (default: int8)
#   ONNX_ASR_MODEL_DIR=<path>           - local weights dir (default: download from HF)
#   ONNX_ASR_CHUNK_SEC=20.0             - max chunk length, long audio split at pauses
#   MODEL_WORKERS=N                     - parallel model instances (default: 1)
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh onnx_asr

export ASR_ENGINE=onnx_asr

# Кэш HuggingFace держим в проекте, а не в ~/.cache - веса до сотен МБ.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data}"

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
