#!/bin/bash
#
# Run Whisper Large V3 using Hugging Face Transformers engine
#
export ASR_ENGINE=transformers
export WHISPER_MODEL=openai/whisper-large-v3
python main.py
