#!/bin/bash
#
# Build OAITT Docker image with GigaAM support
# Supports multi-platform builds (AMD64/ARM64) and pushing to registry
#
# Usage:
#   ./build.sh [options] <image_name> <version>
#
# Examples:
#   ./build.sh myuser/oaitt-gigaam 1.0.0
#   ./build.sh --amd64 myuser/oaitt-gigaam 1.0.0
#   ./build.sh --arm64 myuser/oaitt-gigaam 1.0.0
#   ./build.sh --amd64 --arm64 --push myuser/oaitt-gigaam 1.0.0
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
DOCKERFILE="Dockerfile.cpu"

# Help message
show_help() {
    cat << EOF
Build OAITT Docker image with GigaAM support

Usage:
    $(basename "$0") [OPTIONS] <image_name> <version>

Arguments:
    image_name    Docker image name (e.g., myuser/oaitt-gigaam)
    version       Image version/tag (e.g., 1.0.0, latest)

Options:
    --amd64       Build for AMD64 (x86_64) architecture
    --arm64       Build for ARM64 (aarch64) architecture
    --push        Push image to registry after build
    --file FILE   Use specific Dockerfile (default: Dockerfile.cpu)
    -h, --help    Show this help message

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

Note:
    If no platform specified, builds for current platform only.
    Multi-platform builds require Docker buildx and QEMU emulation.

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
            DOCKERFILE="$2"
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
echo "Image:    $FULL_TAG"
echo "Dockerfile: $DOCKERFILE"
[[ -n "$PLATFORMS" ]] && echo "Platforms: $PLATFORMS"
echo "Push:     $PUSH"
echo ""

# Check if data/gigaam exists
if [[ ! -d "data/gigaam" ]] || [[ -z "$(ls -A data/gigaam 2>/dev/null)" ]]; then
    echo -e "${YELLOW}⚠️  Warning: data/gigaam is empty or missing${NC}"
    echo "Run ./prepare.sh first to download GigaAM models"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

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
    
    BUILD_CMD="$BUILD_CMD ."
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
        BUILD_CMD="$BUILD_CMD ."
    else
        BUILD_CMD="$BUILD_CMD ."
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
        echo "To run the container:"
        echo "  docker run -d -p 9007:9007 $FULL_TAG"
    fi
else
    echo ""
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi
