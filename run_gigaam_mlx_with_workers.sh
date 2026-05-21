#!/bin/bash
#
# Run OAITT with GigaAM-MLX ASR and multiple workers.
# Each worker = separate process с своей копией MLX модели (~850 MB unified memory).
#
# Usage:
#   ./run_gigaam_mlx_with_workers.sh [NUM_WORKERS]
#
# Examples:
#   ./run_gigaam_mlx_with_workers.sh         # 1 worker (default)
#   ./run_gigaam_mlx_with_workers.sh 4       # 4 workers
#   WORKER_MEMORY_LIMIT_MB=4096 ./run_gigaam_mlx_with_workers.sh 4
#
# Note: для MLX каждый worker уже использует in-process lock-free threading.
# Так что MODEL_WORKERS=2 + 4 параллельных клиента = 2 модели x ~8 потоков GPU.
#

NUM_WORKERS=${1:-1}

export ASR_ENGINE=gigaam_mlx
export GIGAAM_MLX_MODEL_TYPE=${GIGAAM_MLX_MODEL_TYPE:-rnnt}
export MODEL_WORKERS=$NUM_WORKERS

# Memory limit per worker (MB) - restart worker if exceeded.
# Default: 6 GB. MLX модель ~850 MB + activations 2-3 GB на параллельные клиенты.
export WORKER_MEMORY_LIMIT_MB=${WORKER_MEMORY_LIMIT_MB:-6144}

# Add vendor/gigaam-mlx to Python path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/vendor/gigaam-mlx:${PYTHONPATH}"

# Auto-detect locally converted weights
LOCAL_MLX_DIR="${SCRIPT_DIR}/data/gigaam_mlx/${GIGAAM_MLX_MODEL_TYPE}"
if [ -z "${GIGAAM_MLX_REPO_ID}" ] && [ -f "${LOCAL_MLX_DIR}/weights.safetensors" ]; then
    export GIGAAM_MLX_REPO_ID="${LOCAL_MLX_DIR}"
    echo "Using locally converted MLX weights: ${LOCAL_MLX_DIR}"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  OAITT with GigaAM-MLX - Worker Pool Mode"
echo "═══════════════════════════════════════════════════════════════"
echo "Workers: $NUM_WORKERS"
echo "Model type: $GIGAAM_MLX_MODEL_TYPE"
echo "Memory limit per worker: ${WORKER_MEMORY_LIMIT_MB} MB"
echo "═══════════════════════════════════════════════════════════════"

python main.py
