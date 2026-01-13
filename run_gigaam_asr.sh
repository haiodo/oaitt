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

export ASR_ENGINE=gigaam
export GIGAAM_MODEL=${GIGAAM_MODEL:-v3_e2e_ctc}

# Add vendor/gigaam to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam:${PYTHONPATH}"

python main.py
