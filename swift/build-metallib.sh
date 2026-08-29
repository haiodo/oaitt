#!/bin/bash
#
# Собирает mlx.metallib из исходников mlx-swift.
#
# Без него бинарь падает с "Failed to load the default metallib": SwiftPM не собирает
# Metal-шейдеры, это делает только Xcode-проект mlx-swift. Раньше файл копировался из
# питоновского wheel mlx - это тянуло зависимость от venv в релизную сборку.
#
# mlx-swift собран в JIT-режиме (jit_kernels.cpp компилируется, nojit_kernels.cpp
# исключён в его Package.swift), поэтому нужен только базовый набор кернелов - тот, что
# в CMakeLists.txt лежит вне блока `if(NOT MLX_METAL_JIT)`. Остальное MLX генерирует и
# компилирует на лету.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$SCRIPT_DIR/.build/checkouts/mlx-swift"
MLX_ROOT="$CHECKOUT/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"
OUTPUT="${1:-$SCRIPT_DIR/.build/release/mlx.metallib}"
DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"

if [ ! -d "$KERNELS" ]; then
    echo "Исходники mlx-swift не найдены. Сначала: swift build -c release" >&2
    exit 1
fi

METAL_VERSION=$(echo "__METAL_VERSION__" | xcrun -sdk macosx metal -E -x metal -P - | tail -1 | tr -d '\n')
echo "==> Metal $METAL_VERSION, deployment target $DEPLOYMENT_TARGET"

KERNEL_NAMES=(
    arg_reduce conv gemv layer_norm random rms_norm rope
    scaled_dot_product_attention
)
# fence.metal собирается только цепочкой, где есть coherent(system) и
# metal::atomic_thread_fence. В Metal Toolchain 26 их нет, и MLX прекрасно работает без
# него - fence нужен для межустройственной синхронизации, которой здесь не бывает.
if [ "$METAL_VERSION" -ge 320 ] && [ "${MLX_BUILD_FENCE:-0}" = "1" ]; then
    KERNEL_NAMES+=(fence)
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for name in "${KERNEL_NAMES[@]}"; do
    printf '    %s\n' "$name"
    xcrun -sdk macosx metal \
        -x metal -Wall -Wextra -fno-fast-math \
        -Wno-c++17-extensions -Wno-c++20-extensions \
        "-mmacosx-version-min=$DEPLOYMENT_TARGET" \
        -c "$KERNELS/$name.metal" -I"$MLX_ROOT" -o "$WORK/$name.air"
done

mkdir -p "$(dirname "$OUTPUT")"
xcrun -sdk macosx metallib "$WORK"/*.air -o "$OUTPUT"
echo "==> $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
