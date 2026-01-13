#!/usr/bin/env bash
set -euo pipefail

# run_gigaam_asr_batch.sh
# Batch transcription using native GigaAM ASR engine
#
# Usage:
#   ./run_gigaam_asr_batch.sh
#   SAMPLES_DIR=samples/private OUTPUT_DIR=private_output ./run_gigaam_asr_batch.sh
#
# Available model names:
#   - v3_e2e_rnnt (default, best quality with punctuation)
#   - v3_e2e_ctc (end-to-end with punctuation)
#   - v3_rnnt, v3_ctc (without punctuation)
#   - v2_rnnt, v2_ctc (older version)
#   - v1_rnnt, v1_ctc (oldest version)
#
# Notes:
#  - This script runs the batch transcription tool (src/batch_transcribe.py)
#    which scans SAMPLES_DIR for audio files and writes outputs to OUTPUT_DIR.
#  - Uses the gigaam package from vendor/gigaam submodule

SAMPLES_DIR="${SAMPLES_DIR:-samples}"
OUTPUT_DIR="${OUTPUT_DIR:-private_output}"

export ASR_ENGINE=gigaam
export GIGAAM_MODEL="${GIGAAM_MODEL:-v3_e2e_ctc}"

echo "GigaAM Native batch transcription"
echo "  Samples dir: ${SAMPLES_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${GIGAAM_MODEL}"
echo ""

# Ensure we run from the repository root (script directory) so the `src` package is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add vendor/gigaam to Python path
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam:${PYTHONPATH:-}"

python -m src.batch_transcribe --input "${SAMPLES_DIR}" --output "${OUTPUT_DIR}"
