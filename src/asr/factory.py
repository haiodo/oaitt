"""
OAITT — Open AI Transformer Transcriber.

Фабрика для создания ASR моделей.
Предоставляет функцию для создания экземпляра ASR модели
на основе конфигурации (переменной окружения ASR_ENGINE).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import TYPE_CHECKING

from src.config import ASR_ENGINE

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

# Supported ASR engines
SUPPORTED_ENGINES = {
    "transformers": "TransformersASR",
    "whisperx": "WhisperXASR",
}


def create_asr_model() -> "ASRModel":
    """
    Создаёт экземпляр ASR модели на основе конфигурации.

    Выбирает реализацию ASR на основе переменной окружения ASR_ENGINE:
    - "transformers": Hugging Face Transformers с Whisper
    - "whisperx": WhisperX с выравниванием слов

    Returns:
        ASRModel: Экземпляр выбранной ASR модели.

    Raises:
        ValueError: Если указан неподдерживаемый ASR engine.

    Example:
        >>> from src.asr import create_asr_model
        >>> model = create_asr_model()
        >>> model.load_model()
        >>> result = model.transcribe(audio, "transcribe", "en", True, "json")
    """
    engine = ASR_ENGINE.lower().strip()

    logger.info(f"Creating ASR model with engine: {engine}")

    if engine == "whisperx":
        from src.asr.whisperx import WhisperXASR

        return WhisperXASR()

    elif engine == "transformers":
        from src.asr.transformers import TransformersASR

        return TransformersASR()

    else:
        supported = ", ".join(f"'{e}'" for e in SUPPORTED_ENGINES.keys())
        raise ValueError(
            f"Unsupported ASR engine: '{ASR_ENGINE}'. "
            f"Supported engines: {supported}"
        )


def get_engine_info(engine: str | None = None) -> dict:
    """
    Возвращает информацию о ASR движке.

    Args:
        engine: Название движка или None для текущего (из конфигурации).

    Returns:
        Словарь с информацией о движке:
        - name: Название движка
        - class: Имя класса реализации
        - description: Описание
        - features: Список особенностей
    """
    engine = engine or ASR_ENGINE

    engine_info = {
        "transformers": {
            "name": "transformers",
            "class": "TransformersASR",
            "description": "Hugging Face Transformers pipeline with Whisper models",
            "features": [
                "Full MPS (Apple Silicon) support",
                "bfloat16/float16 precision",
                "Chunk-based processing for long audio",
                "Batch processing support",
            ],
        },
        "whisperx": {
            "name": "whisperx",
            "class": "WhisperXASR",
            "description": "WhisperX with phoneme-based word alignment",
            "features": [
                "Accurate word-level timestamps",
                "Phoneme-based alignment model",
                "Confidence metrics per word",
                "Multi-language alignment support",
                "Falls back to CPU on MPS devices",
            ],
        },
    }

    return engine_info.get(engine.lower(), {"name": engine, "class": "Unknown"})


def list_supported_engines() -> list[str]:
    """
    Возвращает список поддерживаемых ASR движков.

    Returns:
        Список названий поддерживаемых движков.
    """
    return list(SUPPORTED_ENGINES.keys())
