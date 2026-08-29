#!/bin/bash
#
# Собирает OAITT.app - приложение в строке меню плюс CLI внутри бандла.
#
#   ./build-app.sh [--open]
#
# Модели в бандл не кладутся: они по 850 МБ и качаются при первом запуске.
# mlx.metallib собирается из исходников mlx-swift скриптом build-metallib.sh -
# зависимости от питоновского venv у сборки нет.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build/release"
APP="$SCRIPT_DIR/.build/OAITT.app"
MACOS="$APP/Contents/MacOS"

VERSION="${VERSION:-0.1.0}"

echo "==> Сборка"
swift build -c release --package-path "$SCRIPT_DIR" --product oaitt-swift
swift build -c release --package-path "$SCRIPT_DIR" --product OAITT

echo "==> Бандл"
rm -rf "$APP"
mkdir -p "$MACOS" "$APP/Contents/Resources"

cp "$BUILD_DIR/OAITT" "$MACOS/OAITT"
cp "$BUILD_DIR/oaitt-swift" "$MACOS/oaitt-swift"

echo "==> Metal-шейдеры"
"$SCRIPT_DIR/build-metallib.sh" "$MACOS/mlx.metallib"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>OAITT</string>
    <key>CFBundleDisplayName</key><string>OAITT</string>
    <key>CFBundleIdentifier</key><string>dev.haiodo.oaitt</string>
    <key>CFBundleExecutable</key><string>OAITT</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$VERSION</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSHumanReadableCopyright</key>
    <string>MIT. GigaAM weights (c) GigaChat Team, MLX Swift (c) Apple.</string>
</dict>
</plist>
PLIST

# Ad-hoc подпись: без неё macOS не даст приложению работать даже локально.
codesign --force --deep --sign - "$APP" 2>/dev/null || \
    echo "codesign не сработал - приложение всё ещё запустится локально" >&2

SIZE=$(du -sh "$APP" | cut -f1)
echo "==> Готово: $APP ($SIZE)"

if [ "${1:-}" = "--open" ]; then
    open "$APP"
fi
