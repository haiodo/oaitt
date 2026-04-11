#!/bin/bash
#
# Run OAITT with GigaAM in CPU mode using Docker
#

set -euo pipefail

# Configuration
IMAGE_NAME="oaitt-gigaam:cpu"
CONTAINER_NAME="oaitt-gigaam-cpu"
PORT="${PORT:-9007}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}OAITT GigaAM CPU Docker Runner${NC}"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if image exists, build if not
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo -e "${YELLOW}Docker image not found. Building...${NC}"
    docker build -f Dockerfile.cpu -t "$IMAGE_NAME" .
fi

# Create data directory if not exists
mkdir -p "$DATA_DIR"

# Stop existing container if running
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop "$CONTAINER_NAME" &> /dev/null || true
    docker rm "$CONTAINER_NAME" &> /dev/null || true
fi

# Run container
echo -e "${GREEN}Starting OAITT GigaAM CPU container...${NC}"
echo "  Port: $PORT"
echo "  Data directory: $DATA_DIR"
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:9007" \
    -e "DEVICE=cpu" \
    -e "ASR_ENGINE=gigaam" \
    -e "GIGAAM_MODEL=v3_e2e_rnnt" \
    -e "MODEL_CACHE_DIR=/app/data" \
    -e "MODEL_WORKERS=1" \
    -e "HOST=0.0.0.0" \
    -e "PORT=9007" \
    -v "${DATA_DIR}:/app/data" \
    --restart unless-stopped \
    "$IMAGE_NAME"

echo -e "${GREEN}Container started!${NC}"
echo ""
echo "Wait a moment for model loading, then test with:"
echo "  curl http://localhost:${PORT}/health"
echo ""
echo "View logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "Stop container:"
echo "  docker stop $CONTAINER_NAME"
