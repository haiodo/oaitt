#!/bin/bash
#
# Test Docker build and functionality with sample data
# Uses build.sh for building, then runs container and tests with curl
#

set -euo pipefail

# Configuration
IMAGE_NAME="oaitt-gigaam"
IMAGE_VERSION="test"
CONTAINER_NAME="oaitt-gigaam-test"
PORT=9008
TIMEOUT=120
SAMPLE_FILE="sample-data/Sobolev_Andrey_1_0_00-2_17.ogg"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    docker stop "$CONTAINER_NAME" &>/dev/null || true
    docker rm "$CONTAINER_NAME" &>/dev/null || true
}

trap cleanup EXIT

echo -e "${GREEN}=== OAITT Docker Test ===${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}[1/4] Checking prerequisites...${NC}"

if ! command -v docker &>/dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    exit 1
fi

if [[ ! -f "build.sh" ]]; then
    echo -e "${RED}✗ build.sh not found${NC}"
    exit 1
fi

if [[ ! -f "$SAMPLE_FILE" ]]; then
    echo -e "${RED}✗ Sample file not found: $SAMPLE_FILE${NC}"
    exit 1
fi

if [[ ! -d "data/gigaam" ]] || [[ -z "$(ls -A data/gigaam 2>/dev/null)" ]]; then
    echo -e "${RED}✗ GigaAM models not found. Run ./prepare.sh first${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"
echo ""

# Build Docker image using build.sh
echo -e "${BLUE}[2/4] Building Docker image via build.sh...${NC}"

if ./build.sh "$IMAGE_NAME" "$IMAGE_VERSION"; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo ""

# Start container
echo -e "${BLUE}[3/4] Starting container...${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_VERSION}"
echo "  Container: $CONTAINER_NAME"
echo "  Port: $PORT"

# Remove existing container if exists
docker rm -f "$CONTAINER_NAME" &>/dev/null || true

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
    "${IMAGE_NAME}:${IMAGE_VERSION}"

echo -e "${GREEN}✓ Container started${NC}"
echo ""

# Wait for health check
echo -e "${BLUE}[4/4] Waiting for service to be ready (timeout: ${TIMEOUT}s)...${NC}"

WAITED=0
while [[ $WAITED -lt $TIMEOUT ]]; do
    if curl -s "http://localhost:${PORT}/health" &>/dev/null; then
        echo -e "${GREEN}✓ Service is ready (${WAITED}s)${NC}"
        break
    fi

    # Show progress every 10 seconds
    if [[ $((WAITED % 10)) -eq 0 ]] && [[ $WAITED -gt 0 ]]; then
        echo "  Still waiting... (${WAITED}s)"
    fi

    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ $WAITED -ge $TIMEOUT ]]; then
    echo -e "${RED}✗ Service failed to start within ${TIMEOUT}s${NC}"
    echo ""
    echo "Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50
    exit 1
fi
echo ""

# Run tests
echo -e "${GREEN}=== Running Tests ===${NC}"
echo ""

# Test 1: Health endpoint
echo -e "${BLUE}Test 1: Health check${NC}"
HEALTH_RESPONSE=$(curl -s "http://localhost:${PORT}/health")
if echo "$HEALTH_RESPONSE" | python3 -m json.tool &>/dev/null; then
    echo "$HEALTH_RESPONSE" | python3 -m json.tool
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${YELLOW}⚠ Health response:${NC} $HEALTH_RESPONSE"
fi
echo ""

# Test 2: Transcription (10 runs, measure average time)
echo -e "${BLUE}Test 2: Transcription benchmark (10 runs)${NC}"
echo "  File: $SAMPLE_FILE"
echo ""

NUM_RUNS=10
TOTAL_TIME=0
SUCCESS_COUNT=0

for i in $(seq 1 $NUM_RUNS); do
    echo -e "${BLUE}Run $i/$NUM_RUNS:${NC}"

    START_TIME=$(date +%s%N)

    TRANSCRIPTION_RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST "http://localhost:${PORT}/v1/audio/transcriptions" \
        -H "Authorization: Bearer key" \
        -F "file=@${SAMPLE_FILE}" \
        -F "model=gigaam" \
        -F "response_format=text")

    END_TIME=$(date +%s%N)
    ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    TOTAL_TIME=$((TOTAL_TIME + ELAPSED_MS))

    HTTP_CODE=$(echo "$TRANSCRIPTION_RESPONSE" | tail -1)
    BODY=$(echo "$TRANSCRIPTION_RESPONSE" | sed '$d')

    echo "  HTTP Status: $HTTP_CODE"
    echo "  Time: ${ELAPSED_MS}ms"

    if [[ "$HTTP_CODE" -eq 200 ]]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "  Transcription: \"$BODY\""
        echo -e "  ${GREEN}✓ Success${NC}"
    else
        echo -e "  ${RED}✗ Failed${NC}"
        echo "  Response: $BODY"
    fi
    echo ""
done

# Calculate statistics
if [[ $SUCCESS_COUNT -gt 0 ]]; then
    AVG_TIME=$((TOTAL_TIME / SUCCESS_COUNT))
    TOTAL_SEC=$((TOTAL_TIME / 1000))
    AVG_SEC=$((AVG_TIME / 1000))
    AVG_MS=$((AVG_TIME % 1000))
else
    AVG_TIME=0
    TOTAL_SEC=0
    AVG_SEC=0
    AVG_MS=0
fi

echo -e "${GREEN}=== Benchmark Results ===${NC}"
echo "  Successful: $SUCCESS_COUNT/$NUM_RUNS"
echo "  Total time: ${TOTAL_SEC}s"
echo "  Average time: ${AVG_SEC}.${AVG_MS}s"
echo ""

if [[ $SUCCESS_COUNT -lt $NUM_RUNS ]]; then
    echo -e "${RED}✗ Some transcriptions failed${NC}"
    echo ""
    echo "Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30
    exit 1
fi

# Summary
echo -e "${GREEN}=== All Tests Passed ===${NC}"
echo ""
echo "Image: ${IMAGE_NAME}:${IMAGE_VERSION}"
echo "Container: $CONTAINER_NAME (port $PORT)"
echo ""
echo "To interact manually:"
echo "  Health: curl http://localhost:${PORT}/health"
echo "  Transcribe: curl -X POST http://localhost:${PORT}/v1/audio/transcriptions -H \"Authorization: Bearer key\" -F \"file=@${SAMPLE_FILE}\" -F \"model=gigaam\""
echo "  Logs: docker logs -f $CONTAINER_NAME"
echo ""
echo "Container will be cleaned up automatically on exit"
