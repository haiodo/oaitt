#!/bin/bash
#
# Run OAITT with the GigaAM Multilingual ASR model (native gigaam engine).
#
# Pre-trained on 2M hours across 70+ languages, charwise CTC decoder.
# Best-in-class WER on Russian, Kazakh, Kyrgyz and Uzbek; moderate on English.
# Unlike v3_e2e_*, these models do not produce punctuation.
#
# Model sizes:
#   multilingual_ctc        - 220M encoder (default)
#   multilingual_large_ctc  - 600M encoder, better quality, slower
#
# Env vars:
#   GIGAAM_MODEL=multilingual_ctc|multilingual_large_ctc
#   MODEL_WORKERS=N   - parallel model instances (default: 1)
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_ROOT}/scripts/venv_guard.sh"
require_venv venv ./prepare-gigaam.sh torch

export ASR_ENGINE=gigaam
export GIGAAM_MODEL=${GIGAAM_MODEL:-multilingual_ctc}

export PYTHONPATH="${PROJECT_ROOT}/vendor/gigaam:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
exec "$PYTHON" main.py
