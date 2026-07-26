#!/bin/bash
#
# Run OAITT with WhisperX engine using whisper-large-v3 model
#
# WhisperX lives in its own venv (venv-whisperx) - it pins torch~=2.8.0 which
# conflicts with the torch>=2.11.0 the GigaAM engines need.
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv-whisperx ./prepare-whisperx.sh whisperx

export ASR_ENGINE=whisperx
export WHISPERX_MODEL=${WHISPERX_MODEL:-large-v3}

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
