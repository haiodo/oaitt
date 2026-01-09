#!/usr/bin/env bash
set -euo pipefail

# run_whisper_transformers_batch.sh
# Batch transcription using Whisper via the Transformers pipeline.
#
# This script sets:
#   ASR_ENGINE=transformers
#   WHISPER_MODEL=openai/whisper-large-v3  (Whisper v3-large)
#
# It then runs the batch transcription tool (src/batch_transcribe.py),
# which scans SAMPLES_DIR for audio files and writes outputs to OUTPUT_DIR.
#
# Usage:
#   ./run_whisper_transformers_batch.sh
#   SAMPLES_DIR=samples/private OUTPUT_DIR=private_output ./run_whisper_transformers_batch.sh
#
# Notes:
#  - If you prefer a different model variant, override WHISPER_MODEL:
#      WHISPER_MODEL=ai-some/whisper-v3-large ./run_whisper_transformers_batch.sh
#  - Ensure required Python deps are installed (transformers, torchaudio, etc.)
#  - The batch tool will create per-file .txt/.json and a summary.csv in the output folder.

SAMPLES_DIR="${SAMPLES_DIR:-samples}"
OUTPUT_DIR="${OUTPUT_DIR:-private_output}"

export ASR_ENGINE=transformers
export WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"

echo "Whisper (Transformers) batch transcription"
echo "  Samples dir: ${SAMPLES_DIR}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Model      : ${WHISPER_MODEL}"
echo ""
# Ensure we run from the repository root (script directory) so the `src` package is importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m src.batch_transcribe --input "${SAMPLES_DIR}" --output "${OUTPUT_DIR}"
