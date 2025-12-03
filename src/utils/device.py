"""
OAITT — Open AI Transformer Transcriber.

Утилиты для работы с устройствами вычислений.
Предоставляет функции для определения и управления устройствами
для инференса моделей (CUDA, MPS, CPU).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import gc
import logging

import torch

from src.config import DEVICE

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Определяет лучшее доступное устройство для инференса.

    Порядок приоритета (если DEVICE="auto"):
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU

    Returns:
        torch.device: Выбранное устройство для вычислений.
    """
    if DEVICE != "auto":
        device = DEVICE
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        # Monkey patching for MPS compatibility
        setattr(torch.distributed, "is_initialized", lambda: False)
    else:
        device = "cpu"

    logger.info(f"Selected device: {device}")
    return torch.device(device)


def is_mps_device(device: torch.device) -> bool:
    """
    Проверяет, является ли устройство MPS (Apple Silicon).

    Args:
        device: Устройство для проверки.

    Returns:
        True если устройство MPS, иначе False.
    """
    return device.type == "mps"


def is_cuda_device(device: torch.device) -> bool:
    """
    Проверяет, является ли устройство CUDA (NVIDIA GPU).

    Args:
        device: Устройство для проверки.

    Returns:
        True если устройство CUDA, иначе False.
    """
    return device.type == "cuda"


def clear_memory_cache() -> None:
    """
    Очищает кэш памяти для текущего устройства.

    Вызывает соответствующую функцию очистки для CUDA или MPS,
    а также запускает сборщик мусора Python.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.debug("MPS cache cleared")

    gc.collect()
    logger.debug("Garbage collection completed")


def get_device_info() -> dict:
    """
    Возвращает информацию о доступных устройствах.

    Returns:
        Словарь с информацией о доступных устройствах.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_configured": DEVICE,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info
