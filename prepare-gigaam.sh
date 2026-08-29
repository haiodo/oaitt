#!/bin/bash
#
# Set up the main virtualenv (venv) for the GigaAM engines and download models.
#
# Covers: GigaAM Native, GigaAM MLX, GigaAM/Whisper via Transformers.
# WhisperX needs its own venv - see ./prepare-whisperx.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/venv"

if [[ ! -d "${SCRIPT_DIR}/vendor/gigaam/gigaam" ]]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtualenv at ${VENV_DIR}..."
    python3.13 -m venv "$VENV_DIR"
fi

PYTHON="${VENV_DIR}/bin/python"

echo "Installing dependencies from requirements.txt..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

# gigaam-multilingual-mlx тянет numpy новее, чем терпит numba (нужна gigaam_mlx
# для VAD-чанкования). Возвращаем numpy в закреплённый диапазон - иначе оба
# MLX-движка падают с "Numba needs NumPy 2.4 or less".
"$PYTHON" -m pip install -q "numpy>=2.1.0,<2.3.0"
"$PYTHON" -c "import numpy, numba; print(f'numpy {numpy.__version__} + numba {numba.__version__} OK')"

echo ""
echo "Downloading GigaAM models..."
./prepare.sh

echo ""
echo "Done. Run an engine with:"
echo "    ./run_gigaam_asr.sh        # GigaAM Native"
echo "    ./run_gigaam_mlx_asr.sh    # GigaAM MLX (Apple Silicon)"
echo "    ./run_gigaam.sh            # GigaAM via Transformers"
