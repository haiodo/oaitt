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

# Engine сам зарезолвит локальные MLX веса в data/gigaam_mlx/<type>/.
# При отсутствии - авто-конвертация из data/gigaam/v3_e2e_<type>.ckpt.

echo "═══════════════════════════════════════════════════════════════"
echo "  OAITT with GigaAM-MLX - Worker Pool Mode"
echo "═══════════════════════════════════════════════════════════════"
echo "Workers: $NUM_WORKERS"
echo "Model type: $GIGAAM_MLX_MODEL_TYPE"
echo "Memory limit per worker: ${WORKER_MEMORY_LIMIT_MB} MB"
echo "═══════════════════════════════════════════════════════════════"

python main.py
