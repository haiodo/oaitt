#!/bin/bash
#
export ASR_ENGINE=transformers
export WHISPER_MODEL=ai-sage/GigaAM-v3
export GIGAAM_REVISION=e2e_ctc
python main.py
