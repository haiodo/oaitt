"""
OAITT — Open AI Transformer Transcriber.

Модули ASR (Automatic Speech Recognition).

Содержит:
- base: Абстрактный базовый класс ASRModel
- transformers: Реализация на Hugging Face Transformers
- whisperx: Реализация на WhisperX с выравниванием слов
- factory: Фабрика для создания ASR моделей

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from src.asr.base import ASRModel
from src.asr.transformers import TransformersASR
from src.asr.whisperx import WhisperXASR
from src.asr.gigaam import GigaAMASR
from src.asr.factory import create_asr_model

__all__ = [
    "ASRModel",
    "TransformersASR",
    "WhisperXASR",
    "GigaAMASR",
    "create_asr_model",
]
