#!/bin/bash
#
# Run Whisper Large V3 using Hugging Face Transformers engine
#
# Uses the main venv - same torch/transformers stack as the GigaAM engines.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh transformers

export ASR_ENGINE=transformers
export WHISPER_MODEL=${WHISPER_MODEL:-openai/whisper-large-v3}

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
