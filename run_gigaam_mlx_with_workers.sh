#!/bin/bash
#
# Run OAITT with GigaAM-MLX ASR and multiple workers.
# Each worker = separate process с своей копией MLX модели (~850 MB unified memory).
#
# Когда MLX engine использовать сколько workers:
#
# - 1 worker (default): максимум throughput на 1 модели через lock-free threading
#   (~30-65 req/s на 5s audio с 8 параллельных клиентов).
#   GPU - shared resource, второй worker не ускорит inference, только удвоит память.
#
# - 2-4 workers: только если нужна изоляция (один worker рухнул - сервис жив).
#   Каждый worker = +850 MB unified memory (модель) + ~1-2 GB activations.
#   2 workers разумно при чувствительности к OOM/segfault. Скорость та же.
#
# - >4 workers: не рекомендуется для MLX. Только распыляет память.
#
# Usage:
#   ./run_gigaam_mlx_with_workers.sh         # 1 worker (рекомендовано)
#   ./run_gigaam_mlx_with_workers.sh 2       # 2 workers (HA / изоляция падений)
#   WORKER_MEMORY_LIMIT_MB=4096 ./run_gigaam_mlx_with_workers.sh 2
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

# Pin model cache to project's data/ (engine ищет data/gigaam_mlx/<type>/ внутри MODEL_CACHE_DIR).
# Без абсолютного пути ./data резолвится от cwd запуска - может оказаться пустым.
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${SCRIPT_DIR}/data}"

# Engine сам зарезолвит локальные MLX веса в ${MODEL_CACHE_DIR}/gigaam_mlx/<type>/.
# При отсутствии - авто-конвертация из ${MODEL_CACHE_DIR}/gigaam/v3_e2e_<type>.ckpt.

echo "═══════════════════════════════════════════════════════════════"
echo "  OAITT with GigaAM-MLX - Worker Pool Mode"
echo "═══════════════════════════════════════════════════════════════"
echo "Workers: $NUM_WORKERS"
echo "Model type: $GIGAAM_MLX_MODEL_TYPE"
echo "Memory limit per worker: ${WORKER_MEMORY_LIMIT_MB} MB"
echo "Project dir: ${SCRIPT_DIR}"
echo "Model cache: ${MODEL_CACHE_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# cd в project root - main.py и scripts.convert_gigaam_to_mlx требуют корректного cwd
cd "${SCRIPT_DIR}"
exec python main.py
