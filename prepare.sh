#!/bin/bash
#
# Prepare environment for OAITT with GigaAM
# Downloads GigaAM models and checks dependencies
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
GIGAAM_DIR="${DATA_DIR}/gigaam"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}OAITT GigaAM Preparation Script${NC}"
echo "================================"

# Check if vendor/gigaam exists
if [[ ! -d "${SCRIPT_DIR}/vendor/gigaam/gigaam" ]]; then
    echo -e "${YELLOW}Initializing gigaam submodule...${NC}"
    git submodule update --init --recursive
fi

# Create data directory
mkdir -p "$GIGAAM_DIR"

# Check if models already exist
MODELS_EXIST=true
for model in "v3_e2e_rnnt" "v3_e2e_ctc"; do
    if [[ ! -f "${GIGAAM_DIR}/${model}.ckpt" ]]; then
        MODELS_EXIST=false
        break
    fi
done

if [[ "$MODELS_EXIST" == true ]]; then
    echo -e "${GREEN}GigaAM models already exist in ${GIGAAM_DIR}${NC}"
    ls -lh "$GIGAAM_DIR"
    exit 0
fi

echo -e "${YELLOW}Downloading GigaAM models...${NC}"
echo "This may take a while (approx 900MB)..."

# Download using Python
cd "$SCRIPT_DIR"
python3 << 'EOF'
import os
import sys

# Add vendor/gigaam to path
sys.path.insert(0, "vendor/gigaam")

try:
    import gigaam
    
    download_root = "data/gigaam"
    os.makedirs(download_root, exist_ok=True)
    
    # Download models
    models = ["v3_e2e_rnnt", "v3_e2e_ctc"]
    
    for model_name in models:
        print(f"\nDownloading {model_name}...")
        try:
            model = gigaam.load_model(
                model_name=model_name,
                fp16_encoder=False,
                use_flash=False,
                device="cpu",
                download_root=download_root,
            )
            print(f"✓ {model_name} downloaded successfully")
            del model
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            sys.exit(1)
    
    print("\n✓ All models downloaded successfully!")
    
except ImportError as e:
    print(f"Error: Cannot import gigaam. Make sure vendor/gigaam is initialized: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error downloading models: {e}")
    sys.exit(1)
EOF

echo -e "${GREEN}Preparation complete!${NC}"
echo ""
echo "Downloaded models:"
ls -lh "$GIGAAM_DIR"
