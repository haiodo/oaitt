#!/usr/bin/env bash
set -euo pipefail

# run_whisperx_batch.sh
# Batch transcription using WhisperX engine
#
# Usage:
#   ./run_whisperx_batch.sh
#   SAMPLES_DIR=samples/private OUTPUT_DIR=private_output ./run_whisperx_batch.sh
#
# Notes:
#  - This runs the batch transcription tool (src/batch_transcribe.py)
#    which scans SAMPLES_DIR for audio files and writes outputs to OUTPUT_DIR.
#  - You can override the WhisperX model via the WHISPERX_MODEL env var:
#      WHISPERX_MODEL=large-v3 ./run_whisperx_batch.sh
#    Default here is 'v3-large' (change if you prefer 'large-v3' or another variant).
#
SAMPLES_DIR="${SAMPLES_DIR:-samples}"
OUTPUT_DIR="${OUTPUT_DIR:-private_output}"

export ASR_ENGINE=whisperx
export WHISPERX_MODEL="${WHISPERX_MODEL:-v3-large}"

echo "WhisperX batch transcription"
echo "  Samples dir: ${SAMPLES_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${WHISPERX_MODEL}"
echo ""
# Ensure we run from the repository root (script directory) so the `src` package is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m src.batch_transcribe --input "${SAMPLES_DIR}" --output "${OUTPUT_DIR}"
