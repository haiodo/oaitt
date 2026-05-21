#!/bin/bash
#
# Run OAITT with GigaAM-MLX in CPU mode using Docker (Linux).
#

set -euo pipefail

IMAGE_NAME="oaitt-gigaam-mlx:cpu"
CONTAINER_NAME="oaitt-gigaam-mlx-cpu"
PORT="${PORT:-9008}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}OAITT GigaAM-MLX CPU Docker Runner${NC}"
echo "===================================="

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Image not found. Building...${NC}"
    docker build -f Dockerfile.mlx-cpu -t "$IMAGE_NAME" .
fi

mkdir -p "$DATA_DIR"

if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop "$CONTAINER_NAME" &> /dev/null || true
    docker rm "$CONTAINER_NAME" &> /dev/null || true
fi

echo -e "${GREEN}Starting container on port ${PORT}...${NC}"

docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:9007" \
    -e "DEVICE=cpu" \
    -e "ASR_ENGINE=gigaam_mlx" \
    -e "GIGAAM_MLX_MODEL_TYPE=${GIGAAM_MLX_MODEL_TYPE:-rnnt}" \
    -e "GIGAAM_MLX_LOCK_FREE=true" \
    -e "MODEL_CACHE_DIR=/app/data" \
    -e "MODEL_WORKERS=${MODEL_WORKERS:-1}" \
    -e "HOST=0.0.0.0" \
    -e "PORT=9007" \
    -v "${DATA_DIR}:/app/data" \
    --restart unless-stopped \
    "$IMAGE_NAME"

echo -e "${GREEN}Container started.${NC}"
echo "  curl http://localhost:${PORT}/health"
echo "  docker logs -f $CONTAINER_NAME"
echo "  docker stop $CONTAINER_NAME"
