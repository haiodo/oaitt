#!/bin/bash
#
# Run OAITT with GigaAM ASR and multiple workers
# Each worker runs in a separate process with automatic restart on memory limit
#
# Usage:
#   ./run_gigaam_with_workers.sh [NUM_WORKERS]
#
# Examples:
#   # 1 worker (default, fastest for single requests)
#   ./run_gigaam_with_workers.sh
#
#   # 4 workers with 7GB memory limit per worker
#   ./run_gigaam_with_workers.sh 4
#
#   # 4 workers with custom memory limit (5GB)
#   WORKER_MEMORY_LIMIT_MB=5120 ./run_gigaam_with_workers.sh 4
#

NUM_WORKERS=${1:-1}

export ASR_ENGINE=gigaam
export GIGAAM_MODEL=${GIGAAM_MODEL:-v3_e2e_ctc}
export MODEL_WORKERS=$NUM_WORKERS

# Memory limit per worker (MB) - restart worker if exceeded
# Default: 7GB (7168 MB) - adjust based on your system
# Set to 0 to disable memory-based restart
export WORKER_MEMORY_LIMIT_MB=${WORKER_MEMORY_LIMIT_MB:-7168}

# Add vendor/gigaam to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam:${PYTHONPATH}"

echo "═══════════════════════════════════════════════════════════════"
echo "  OAITT with GigaAM - Worker Pool Mode"
echo "═══════════════════════════════════════════════════════════════"
echo "Workers: $NUM_WORKERS"
echo "Memory limit per worker: ${WORKER_MEMORY_LIMIT_MB} MB"
echo "Auto-restart on memory limit: $([ "$WORKER_MEMORY_LIMIT_MB" -gt 0 ] && echo 'enabled' || echo 'disabled')"
echo "═══════════════════════════════════════════════════════════════"

python main.py
