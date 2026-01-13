#!/bin/bash
#
# Run OAITT with WhisperX engine using whisper-large-v3 model
#
export ASR_ENGINE=whisperx
export WHISPERX_MODEL=large-v3
python main.py
