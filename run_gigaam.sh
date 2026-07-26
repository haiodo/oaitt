#!/bin/bash
#
# Run GigaAM-v3 via the Hugging Face Transformers engine.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh transformers

export ASR_ENGINE=transformers
export WHISPER_MODEL=${WHISPER_MODEL:-ai-sage/GigaAM-v3}
export GIGAAM_REVISION=${GIGAAM_REVISION:-e2e_ctc}

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
