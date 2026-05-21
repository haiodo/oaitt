"""
OAITT — Open AI Transformer Transcriber.

Модули ASR (Automatic Speech Recognition).

Содержит:
- base: Абстрактный базовый класс ASRModel
- transformers: Реализация на Hugging Face Transformers
- whisperx: Реализация на WhisperX с выравниванием слов
- gigaam: PyTorch GigaAM
- gigaam_mlx: MLX GigaAM (без PyTorch)
- factory: Фабрика для создания ASR моделей

Реализации движков импортируются лениво через factory.create_asr_model(),
чтобы образы без PyTorch (gigaam_mlx CPU) не падали при `from src.asr import ...`.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from src.asr.base import ASRModel
from src.asr.factory import create_asr_model

__all__ = [
    "ASRModel",
    "create_asr_model",
]
