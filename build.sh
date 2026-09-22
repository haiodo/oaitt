#!/bin/bash
#
# Build OAITT Docker image with GigaAM support
# Supports multi-platform builds (AMD64/ARM64) and pushing to registry
#
# Usage:
#   ./build.sh [options] <image_name> <version>
#   ./build.sh                       # interactive wizard
#
# Examples:
#   ./build.sh myuser/oaitt-gigaam 1.0.0
#   ./build.sh --amd64 myuser/oaitt-gigaam 1.0.0
#   ./build.sh --arm64 myuser/oaitt-gigaam 1.0.0
#   ./build.sh --amd64 --arm64 --push myuser/oaitt-gigaam 1.0.0
#   ./build.sh --target onnx myuser/oaitt-onnx 1.0.0
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_AMD64=false
BUILD_ARM64=false
PUSH=false
PLATFORMS=""
IMAGE_NAME=""
VERSION=""
DOCKERFILE_OVERRIDE=""
DOCKERFILE=""
BUILD_CONTEXT="."
TARGET=""
INTERACTIVE=false
LAST_FILE="$(dirname "${BASH_SOURCE[0]}")/.build.last"

# Таблица целей сборки: имя -> Dockerfile, build context, проверяемые директории data/.
TARGETS="onnx cpu mlx-cpu models"

target_dockerfile() {
    case "$1" in
        onnx) echo "Dockerfile.onnx" ;;
        cpu) echo "Dockerfile.cpu" ;;
        mlx-cpu) echo "Dockerfile.mlx-cpu" ;;
        models) echo "Dockerfile.models" ;;
        *) return 1 ;;
    esac
}

target_context() {
    case "$1" in
        onnx|cpu|mlx-cpu) echo "." ;;
        models) echo "data" ;;
        *) return 1 ;;
    esac
}

# Список директорий data/, которые должны быть непустыми для цели (пусто - проверка не нужна).
target_prereqs() {
    case "$1" in
        onnx) echo "" ;;
        cpu) echo "data/gigaam" ;;
        mlx-cpu) echo "data/gigaam_mlx" ;;
        models) echo "data/gigaam_mlx data/parakeet_tdt_v3 data/hub/models--mlx-community--parakeet-tdt-0.6b-v3" ;;
        *) return 1 ;;
    esac
}

default_image_for_target() {
    case "$1" in
        onnx) echo "myuser/oaitt-onnx" ;;
        cpu) echo "myuser/oaitt-gigaam" ;;
        mlx-cpu) echo "myuser/oaitt-gigaam-mlx" ;;
        models) echo "myuser/oaitt-models" ;;
    esac
}

# Реестр, в который уйдёт push: хост перед первым "/", если он похож на хост, иначе Docker Hub.
registry_of() {
    local img="$1"
    local first="${img%%/*}"
    if [[ "$img" == */* ]] && [[ "$first" == *.* || "$first" == *:* || "$first" == "localhost" ]]; then
        echo "$first"
    else
        echo "docker.io"
    fi
}

check_prereqs() {
    local target="$1"
    local dirs
    if [[ -n "$target" ]]; then
        dirs=$(target_prereqs "$target")
    else
        dirs="data/gigaam"
    fi
    [[ -z "$dirs" ]] && return 0

    local missing=()
    local d
    for d in $dirs; do
        if [[ ! -d "$d" ]] || [[ -z "$(ls -A "$d" 2>/dev/null)" ]]; then
            missing+=("$d")
        fi
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0

    echo -e "${YELLOW}⚠️  Warning: empty or missing: ${missing[*]}${NC}"
    echo "Run ./prepare.sh first to download models"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

run_interactive() {
    echo -e "${GREEN}=== OAITT Docker Build (interactive) ===${NC}"
    echo ""

    echo "Select build target:"
    local i=1 t
    for t in $TARGETS; do
        printf "  %d) %-10s %s\n" "$i" "$t" "$(target_dockerfile "$t")"
        i=$((i + 1))
    done
    local n_targets
    n_targets=$(wc -w <<< "$TARGETS")
    local choice
    while true; do
        read -rp "Target [1]: " choice
        choice="${choice:-1}"
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= n_targets )); then
            TARGET=$(cut -d' ' -f"$choice" <<< "$TARGETS")
            break
        fi
        echo -e "${RED}Invalid choice${NC}"
    done

    local default_image last_image=""
    default_image=$(default_image_for_target "$TARGET")
    [[ -f "$LAST_FILE" ]] && last_image=$(grep "^IMAGE_NAME=" "$LAST_FILE" 2>/dev/null | cut -d= -f2-)
    [[ -n "$last_image" ]] && default_image="$last_image"
    read -rp "Image name (registry included) [$default_image]: " IMAGE_NAME
    IMAGE_NAME="${IMAGE_NAME:-$default_image}"

    read -rp "Tag [latest]: " VERSION
    VERSION="${VERSION:-latest}"

    echo "Platforms:"
    echo "  1) current (default)"
    echo "  2) amd64"
    echo "  3) arm64"
    echo "  4) both"
    while true; do
        read -rp "Choice [1]: " choice
        choice="${choice:-1}"
        case "$choice" in
            1) BUILD_AMD64=false; BUILD_ARM64=false; break ;;
            2) BUILD_AMD64=true; BUILD_ARM64=false; break ;;
            3) BUILD_AMD64=false; BUILD_ARM64=true; break ;;
            4) BUILD_AMD64=true; BUILD_ARM64=true; break ;;
            *) echo -e "${RED}Invalid choice${NC}" ;;
        esac
    done

    while true; do
        read -rp "Push to registry? (y/N): " choice
        choice="${choice:-n}"
        case "$choice" in
            [Yy]*) PUSH=true; break ;;
            [Nn]*) PUSH=false; break ;;
            *) echo -e "${RED}Please answer y or n${NC}" ;;
        esac
    done

    echo "IMAGE_NAME=$IMAGE_NAME" > "$LAST_FILE"
    echo ""
}

if [[ $# -eq 0 ]]; then
    INTERACTIVE=true
    run_interactive
fi

# Help message
show_help() {
    cat << EOF
Build OAITT Docker image with GigaAM support

Usage:
    $(basename "$0") [OPTIONS] <image_name> <version>
    $(basename "$0")                            # interactive wizard

Arguments:
    image_name    Docker image name (e.g., myuser/oaitt-gigaam)
    version       Image version/tag (e.g., 1.0.0, latest)

Options:
    --amd64          Build for AMD64 (x86_64) architecture
    --arm64          Build for ARM64 (aarch64) architecture
    --push           Push image to registry after build
    --target NAME    Build target: onnx, cpu, mlx-cpu, models (alternative to --file)
    --file FILE      Use specific Dockerfile (default: Dockerfile.cpu)
    -h, --help       Show this help message

Examples:
    # Build for local platform only
    $(basename "$0") myuser/oaitt-gigaam 1.0.0

    # Build for AMD64 only
    $(basename "$0") --amd64 myuser/oaitt-gigaam 1.0.0

    # Build for both platforms
    $(basename "$0") --amd64 --arm64 myuser/oaitt-gigaam 1.0.0

    # Build and push multi-platform image
    $(basename "$0") --amd64 --arm64 --push myuser/oaitt-gigaam 1.0.0

    # Build with custom Dockerfile
    $(basename "$0") --file Dockerfile.cpu --amd64 --push myuser/oaitt-gigaam latest

    # Build a named target
    $(basename "$0") --target onnx myuser/oaitt-onnx 1.0.0

Note:
    If no platform specified, builds for current platform only.
    Multi-platform builds require Docker buildx and QEMU emulation.
    Running with no arguments at all starts an interactive wizard.

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --amd64)
            BUILD_AMD64=true
            shift
            ;;
        --arm64)
            BUILD_ARM64=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --file)
            DOCKERFILE_OVERRIDE="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # First positional argument is image name, second is version
            if [[ -z "$IMAGE_NAME" ]]; then
                IMAGE_NAME="$1"
            elif [[ -z "$VERSION" ]]; then
                VERSION="$1"
            else
                echo -e "${RED}Error: Too many arguments${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$IMAGE_NAME" ]] || [[ -z "$VERSION" ]]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $(basename "$0") [options] <image_name> <version>"
    echo "Use --help for more information"
    exit 1
fi

# Resolve Dockerfile + build context from --target (falls back to --file / legacy default)
if [[ -n "$TARGET" ]]; then
    if ! DOCKERFILE_FROM_TARGET=$(target_dockerfile "$TARGET"); then
        echo -e "${RED}Error: unknown target '$TARGET' (expected one of: $TARGETS)${NC}"
        exit 1
    fi
    DOCKERFILE="${DOCKERFILE_OVERRIDE:-$DOCKERFILE_FROM_TARGET}"
    BUILD_CONTEXT=$(target_context "$TARGET")
else
    DOCKERFILE="${DOCKERFILE_OVERRIDE:-Dockerfile.cpu}"
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    echo -e "${RED}Error: Dockerfile not found: $DOCKERFILE${NC}"
    exit 1
fi

# Build platforms string
if [[ "$BUILD_AMD64" == true ]] && [[ "$BUILD_ARM64" == true ]]; then
    PLATFORMS="linux/amd64,linux/arm64"
    echo -e "${BLUE}Building for multi-platform: AMD64 + ARM64${NC}"
elif [[ "$BUILD_AMD64" == true ]]; then
    PLATFORMS="linux/amd64"
    echo -e "${BLUE}Building for platform: AMD64${NC}"
elif [[ "$BUILD_ARM64" == true ]]; then
    PLATFORMS="linux/arm64"
    echo -e "${BLUE}Building for platform: ARM64${NC}"
else
    echo -e "${BLUE}Building for current platform only${NC}"
fi

# Full image tag
FULL_TAG="${IMAGE_NAME}:${VERSION}"

echo -e "${GREEN}=== OAITT Docker Build ===${NC}"
[[ -n "$TARGET" ]] && echo "Target:   $TARGET"
echo "Image:    $FULL_TAG"
echo "Dockerfile: $DOCKERFILE"
echo "Context:  $BUILD_CONTEXT"
[[ -n "$PLATFORMS" ]] && echo "Platforms: $PLATFORMS"
echo "Push:     $PUSH"
# docker login не проверить надёжно - просто показываем, куда уйдёт push.
[[ "$PUSH" == true ]] && echo "Registry: $(registry_of "$IMAGE_NAME")"
echo ""

if [[ "$INTERACTIVE" == true ]]; then
    read -rp "Proceed? (Y/n): " CONFIRM
    CONFIRM="${CONFIRM:-y}"
    if [[ ! "$CONFIRM" =~ ^[Yy] ]]; then
        echo "Aborted"
        exit 0
    fi
fi

check_prereqs "$TARGET"

# Setup buildx for multi-platform builds
if [[ -n "$PLATFORMS" ]]; then
    echo -e "${BLUE}Setting up Docker buildx...${NC}"
    
    # Check if buildx is available
    if ! docker buildx version &>/dev/null; then
        echo -e "${RED}Error: Docker buildx not available${NC}"
        echo "Please install Docker Desktop or enable buildx plugin"
        exit 1
    fi
    
    # Create a new builder instance if not exists
    BUILDER_NAME="oaitt-builder"
    if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null 2>&1; then
        echo "Creating buildx builder: $BUILDER_NAME"
        docker buildx create --name "$BUILDER_NAME" --driver docker-container --bootstrap
    fi
    
    # Use the builder
    docker buildx use "$BUILDER_NAME"
    
    # Inspect builder
    docker buildx inspect --bootstrap
    
    echo ""
fi

# Build command construction
echo -e "${GREEN}Starting build...${NC}"

if [[ -n "$PLATFORMS" ]]; then
    # Multi-platform build
    BUILD_CMD="docker buildx build"
    BUILD_CMD="$BUILD_CMD --platform $PLATFORMS"
    BUILD_CMD="$BUILD_CMD --tag $FULL_TAG"
    BUILD_CMD="$BUILD_CMD --file $DOCKERFILE"
    
    if [[ "$PUSH" == true ]]; then
        BUILD_CMD="$BUILD_CMD --push"
        echo -e "${BLUE}Building and pushing multi-platform image...${NC}"
    else
        BUILD_CMD="$BUILD_CMD --load"
        echo -e "${BLUE}Building multi-platform image (local load)...${NC}"
        echo -e "${YELLOW}Note: --load only works for single platform. Use --push for multi-platform.${NC}"
    fi

    BUILD_CMD="$BUILD_CMD $BUILD_CONTEXT"
else
    # Single platform build
    BUILD_CMD="docker build"
    BUILD_CMD="$BUILD_CMD --tag $FULL_TAG"
    BUILD_CMD="$BUILD_CMD --file $DOCKERFILE"
    
    if [[ "$PUSH" == true ]]; then
        # For single platform with push, we still use buildx for consistency
        BUILD_CMD="docker buildx build --push"
        BUILD_CMD="$BUILD_CMD --tag $FULL_TAG"
        BUILD_CMD="$BUILD_CMD --file $DOCKERFILE"
        BUILD_CMD="$BUILD_CMD $BUILD_CONTEXT"
    else
        BUILD_CMD="$BUILD_CMD $BUILD_CONTEXT"
    fi
fi

echo "Command: $BUILD_CMD"
echo ""

# Execute build
if eval "$BUILD_CMD"; then
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    echo "Image: $FULL_TAG"
    
    if [[ "$PUSH" == true ]]; then
        echo -e "${GREEN}✅ Image pushed to registry${NC}"
    else
        # Show image info
        docker images "$IMAGE_NAME" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -5
        echo ""
        echo "To push to registry, run:"
        echo "  docker push $FULL_TAG"
        echo ""
        if [[ "$TARGET" == "models" ]]; then
            echo "To export the weights:"
            echo "  docker run --rm -v \"\$PWD/data:/out\" $FULL_TAG"
        else
            echo "To run the container:"
            echo "  docker run -d -p 9007:9007 $FULL_TAG"
        fi
    fi
else
    echo ""
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
