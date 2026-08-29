#!/bin/bash
#
# Run the native Swift/MLX GigaAM server (Apple Silicon).
#
# Env vars:
#   GIGAAM_MLX_MODEL_TYPE=ctc|rnnt   - model variant (default: rnnt)
#   PORT=9007                        - listen port
#   AUTH_TOKEN=key                   - bearer token, как у Python-версии; пусто - без авторизации
#   BUILD_CONFIG=release|debug       - swift build configuration (default: release)
#   ASR_CACHE_SIZE=256               - размер кеша результатов; 0 отключает
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TYPE=${GIGAAM_MLX_MODEL_TYPE:-rnnt}
PORT=${PORT:-9007}
BUILD_CONFIG=${BUILD_CONFIG:-release}
BIN_DIR="${SCRIPT_DIR}/swift/.build/${BUILD_CONFIG}"

swift build -c "$BUILD_CONFIG" --package-path "${SCRIPT_DIR}/swift"

# SwiftPM не собирает Metal-шейдеры mlx-swift, а без них бинарь падает с
# "Failed to load the default metallib". MLX ищет colocated mlx.metallib первым.
if [ ! -f "${BIN_DIR}/mlx.metallib" ]; then
    "${SCRIPT_DIR}/swift/build-metallib.sh" "${BIN_DIR}/mlx.metallib"
fi

exec "${BIN_DIR}/oaitt-swift" serve \
    --host "${HOST:-127.0.0.1}" \
    --port "$PORT" \
    --model-cache-dir "${SCRIPT_DIR}/data/gigaam_mlx" \
    --model-type "$MODEL_TYPE" \
    --cache-size "${ASR_CACHE_SIZE-256}" \
    --api-key "${AUTH_TOKEN-key}"
