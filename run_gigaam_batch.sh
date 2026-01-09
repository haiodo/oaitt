#!/usr/bin/env bash
set -euo pipefail

# run_gigaam_batch.sh
# Batch transcription using GigaAM via Transformers (trust_remote_code=True)
#
# Usage:
#   ./run_gigaam_batch.sh
#   SAMPLES_DIR=samples/private OUTPUT_DIR=private_output ./run_gigaam_batch.sh
#
# Notes:
#  - This script runs the batch transcription tool (src/batch_transcribe.py)
#    which scans SAMPLES_DIR for audio files and writes outputs to OUTPUT_DIR.
#  - Ensure necessary dependencies for GigaAM are installed if using HF repo
#    (hydra-core, omegaconf, torchaudio, sentencepiece, ...), or install the
#    GigaAM package and use ASR_ENGINE=gigaam instead.

SAMPLES_DIR="${SAMPLES_DIR:-samples}"
OUTPUT_DIR="${OUTPUT_DIR:-private_output}"

export ASR_ENGINE=transformers
export WHISPER_MODEL=ai-sage/GigaAM-v3
export GIGAAM_REVISION="${GIGAAM_REVISION:-e2e_ctc}"

echo "GigaAM batch transcription"
echo "  Samples dir: ${SAMPLES_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${WHISPER_MODEL} (revision=${GIGAAM_REVISION})"
echo ""

# Ensure we run from the repository root (script directory) so the `src` package is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m src.batch_transcribe --input "${SAMPLES_DIR}" --output "${OUTPUT_DIR}"
