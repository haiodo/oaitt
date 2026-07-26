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
import os
from typing import Optional, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    MLX_AVAILABLE = False

from src.config import DEVICE

logger = logging.getLogger(__name__)


def get_device() -> Any:
    """
    Определяет лучшее доступное устройство для инференса.

    Порядок приоритета (если DEVICE="auto"):
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU

    Returns:
        torch.device или строка "cpu" если torch недоступен.
    """
    if not TORCH_AVAILABLE:
        logger.info("torch unavailable, returning 'cpu' string")
        return "cpu"

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


def is_mps_device(device: Any) -> bool:
    """
    Проверяет, является ли устройство MPS (Apple Silicon).

    Args:
        device: Устройство для проверки.

    Returns:
        True если устройство MPS, иначе False.
    """
    if not TORCH_AVAILABLE or isinstance(device, str):
        return False
    return device.type == "mps"


def is_cuda_device(device: Any) -> bool:
    """
    Проверяет, является ли устройство CUDA (NVIDIA GPU).

    Args:
        device: Устройство для проверки.

    Returns:
        True если устройство CUDA, иначе False.
    """
    if not TORCH_AVAILABLE or isinstance(device, str):
        return False
    return device.type == "cuda"


def clear_memory_cache() -> None:
    """
    Очищает кэш памяти для текущего устройства.

    Вызывает соответствующую функцию очистки для CUDA или MPS.
    """
    if not TORCH_AVAILABLE:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("CUDA cache cleared and synchronized")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        # MPS synchronize may not be available in all PyTorch versions
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
        logger.debug("MPS cache cleared")


def get_device_info() -> dict:
    """
    Возвращает информацию о доступных устройствах.

    Returns:
        Словарь с информацией о доступных устройствах.
    """
    if not TORCH_AVAILABLE:
        return {
            "cuda_available": False,
            "mps_available": False,
            "device_configured": DEVICE,
            "torch_available": False,
        }

    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_configured": DEVICE,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def get_process_memory_mb() -> float:
    """
    Возвращает текущее использование памяти процессом в МБ.

    Использует psutil для получения RSS (Resident Set Size).

    Returns:
        Использование памяти в МБ, или -1 если psutil недоступен.
    """
    if not PSUTIL_AVAILABLE:
        return -1.0

    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger.warning(f"Failed to get process memory: {e}")
        return -1.0


def get_gpu_memory_mb() -> Optional[float]:
    """
    Возвращает использование памяти GPU в МБ.

    Для MLX возвращает active memory собственного аллокатора MLX.
    Для CUDA возвращает allocated memory.
    Для MPS возвращает allocated memory (если доступно).

    Returns:
        Использование памяти GPU в МБ, или None если GPU недоступен.
    """
    # MLX держит буферы в своём аллокаторе. torch.mps.driver_allocated_memory()
    # видит их как часть unified memory процесса, но вместе с torch-буферами -
    # для MLX-движка спрашиваем MLX напрямую.
    # active - буферы, живые прямо сейчас (веса + текущие активации).
    # Кэш аллокатора сюда не входит намеренно: это переиспользуемые свободные
    # буферы, они не отражаются в RSS и растут до cache_limit независимо от
    # реального потребления. Пик см. в gpu_memory_peak_mb.
    if MLX_AVAILABLE:
        try:
            active = mx.get_active_memory() / (1024 * 1024)
            if active > 0:
                return active
        except Exception as e:
            logger.debug(f"MLX memory query failed: {e}")

    if not TORCH_AVAILABLE:
        return None

    if torch.cuda.is_available():
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get CUDA memory: {e}")
            return None

    if torch.backends.mps.is_available():
        try:
            # MPS doesn't have direct memory query, but we can get driver allocated
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception as e:
            logger.debug(f"MPS memory query not available: {e}")
            return None

    return None


def get_memory_info() -> dict:
    """
    Возвращает полную информацию об использовании памяти.

    Returns:
        Словарь с информацией о памяти:
        - process_memory_mb: Память процесса (RSS) в МБ
        - gpu_memory_mb: Память GPU в МБ (если доступно)
        - gpu_memory_reserved_mb: Зарезервированная память GPU (CUDA only)
        - system_memory_total_mb: Общая системная память
        - system_memory_available_mb: Доступная системная память
    """
    info = {
        "process_memory_mb": round(get_process_memory_mb(), 1),
    }

    # GPU memory
    gpu_mem = get_gpu_memory_mb()
    if gpu_mem is not None:
        info["gpu_memory_mb"] = round(gpu_mem, 1)

    # Peak GPU memory - для планирования RAM пиковое значение важнее текущего.
    if MLX_AVAILABLE:
        try:
            peak = mx.get_peak_memory()
            if peak > 0:
                info["gpu_memory_peak_mb"] = round(peak / (1024 * 1024), 1)
                info["gpu_memory_cache_mb"] = round(
                    mx.get_cache_memory() / (1024 * 1024), 1
                )
        except Exception:
            pass

    # CUDA reserved memory
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            info["gpu_memory_reserved_mb"] = round(
                torch.cuda.memory_reserved() / (1024 * 1024), 1
            )
            info["gpu_memory_peak_mb"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 1
            )
        except Exception:
            pass

    # System memory
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            info["system_memory_total_mb"] = round(mem.total / (1024 * 1024), 1)
            info["system_memory_available_mb"] = round(mem.available / (1024 * 1024), 1)
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")

    return info


def get_model_memory_usage(baseline_mb: float) -> float:
    """
    Вычисляет использование памяти моделью относительно базовой линии.

    Args:
        baseline_mb: Базовое использование памяти до загрузки модели в МБ.

    Returns:
        Разница в использовании памяти в МБ.
    """
    current = get_process_memory_mb()
    if current < 0 or baseline_mb < 0:
        return -1.0
    return current - baseline_mb
