#!/bin/bash
#
# Run OAITT with native GigaAM ASR engine
# Uses the gigaam package from vendor/gigaam submodule
#
# Available model names:
#   - v3_e2e_rnnt (default, best quality with punctuation)
#   - v3_e2e_ctc (end-to-end with punctuation)
#   - v3_rnnt, v3_ctc (without punctuation)
#   - v2_rnnt, v2_ctc (older version)
#   - v1_rnnt, v1_ctc (oldest version)
#
# Parallel inference options:
#   MODEL_WORKERS=N             - Number of model instances (default: 1)
#                                 Each worker loads its own copy of the model
#                                 and processes requests independently.
#                                 Example: MODEL_WORKERS=4 ./run_gigaam_asr.sh
#
# Memory debugging options:
#   MEMORY_LOG_ENABLED=true     - Enable periodic memory logging
#   MEMORY_LOG_INTERVAL=60      - Log interval in seconds (default: 60)
#   MEMORY_LOG_TOP_ALLOCATIONS=5 - Number of top allocation sites to log
#
# Example with memory monitoring:
#   MEMORY_LOG_ENABLED=true MEMORY_LOG_INTERVAL=30 ./run_gigaam_asr.sh
#

export ASR_ENGINE=gigaam
export GIGAAM_MODEL=${GIGAAM_MODEL:-v3_e2e_ctc}

# Add vendor/gigaam to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam:${PYTHONPATH}"

python main.py
