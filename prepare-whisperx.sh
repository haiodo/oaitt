#!/bin/bash
#
# Set up a separate virtualenv (venv-whisperx) for the WhisperX engine.
#
# WhisperX pins torch~=2.8.0, while GigaAM needs torch>=2.11.0 for the MPS
# memory-leak fixes on Apple Silicon. Both cannot live in one environment,
# so WhisperX gets its own.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/venv-whisperx"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating virtualenv at ${VENV_DIR}..."
    python3 -m venv "$VENV_DIR"
fi

PYTHON="${VENV_DIR}/bin/python"

echo "Installing WhisperX dependencies..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements-whisperx.txt

# The service itself needs the web stack too - requirements-whisperx.txt only
# covers the engine.
"$PYTHON" -m pip install fastapi uvicorn pydantic python-multipart soundfile librosa psutil

echo ""
echo "Verifying installation..."
"$PYTHON" -c "import whisperx, torch; print(f'whisperx OK, torch {torch.__version__}')"

echo ""
echo "Done. Run it with:"
echo "    ./run_whisperx_large_v3.sh"
