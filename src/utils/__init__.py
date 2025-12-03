"""
OAITT — Open AI Transformer Transcriber.

Утилитарные модули.

Содержит:
- audio: Функции для загрузки и обработки аудио
- device: Функции для работы с устройствами (CUDA, MPS, CPU)
- formatters: Функции форматирования вывода (VTT, SRT, TSV)

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from src.utils.audio import load_audio_from_file
from src.utils.device import get_device, is_mps_device, clear_memory_cache
from src.utils.formatters import (
    format_vtt,
    format_srt,
    format_tsv,
    format_timestamp_vtt,
    format_timestamp_srt,
)

__all__ = [
    "load_audio_from_file",
    "get_device",
    "is_mps_device",
    "clear_memory_cache",
    "format_vtt",
    "format_srt",
    "format_tsv",
    "format_timestamp_vtt",
    "format_timestamp_srt",
]
