"""
OAITT — Open AI Transformer Transcriber.

Маршрут проверки здоровья сервиса.
Предоставляет эндпоинт для проверки статуса сервиса,
включая информацию о загруженной модели и производительности.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter

from src.config import ASR_ENGINE, TIMEOUT_ENABLED
from src.services.memory_monitor import memory_monitor
from src.services.performance import performance_tracker
from src.utils.device import get_memory_info

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Reference to global ASR model (will be set by app)
_asr_model: "ASRModel | None" = None


def set_asr_model(model: "ASRModel") -> None:
    """
    Устанавливает ссылку на глобальную ASR модель.

    Args:
        model: Экземпляр ASR модели.
    """
    global _asr_model
    _asr_model = model


@router.get("/health")
async def health_check() -> dict:
    """
    Проверяет здоровье сервиса.

    Возвращает информацию о:
    - Статусе сервиса
    - Загружена ли модель
    - Используемый ASR движок
    - Статистику производительности
    - Настройки таймаута

    Returns:
        dict: Информация о состоянии сервиса.

    Example response:
        {
            "status": "healthy",
            "model_loaded": true,
            "engine": "whisperx",
            "timeout_enabled": true,
            "performance": {
                "samples": 42,
                "avg_ratio": 0.0853,
                "avg_speed": 11.72,
                "min_ratio": 0.0612,
                "max_ratio": 0.1234
            }
        }
    """
    perf_stats = performance_tracker.get_stats()
    memory_info = get_memory_info()

    # Add model-specific memory if available
    if _asr_model is not None:
        model_info = _asr_model.get_info()
        if "memory" in model_info and "model_memory_mb" in model_info["memory"]:
            memory_info["model_memory_mb"] = model_info["memory"]["model_memory_mb"]

    return {
        "status": "healthy",
        "model_loaded": _asr_model is not None and _asr_model.is_loaded(),
        "engine": ASR_ENGINE,
        "timeout_enabled": TIMEOUT_ENABLED,
        "performance": perf_stats,
        "memory": memory_info,
    }


@router.get("/health/detailed")
async def health_check_detailed() -> dict:
    """
    Возвращает детальную информацию о здоровье сервиса.

    Включает дополнительную информацию о модели, устройстве
    и отладочных логах.

    Returns:
        dict: Детальная информация о состоянии сервиса.
    """
    from src.services.debug import get_debug_log_stats
    from src.utils.device import get_device_info

    perf_stats = performance_tracker.get_stats()
    device_info = get_device_info()
    debug_stats = get_debug_log_stats()

    model_info = {}
    if _asr_model is not None:
        model_info = _asr_model.get_info()

    return {
        "status": "healthy",
        "engine": ASR_ENGINE,
        "timeout_enabled": TIMEOUT_ENABLED,
        "model": model_info,
        "device": device_info,
        "performance": perf_stats,
        "debug_logging": debug_stats,
    }


@router.post("/admin/memory-snapshot")
async def save_memory_snapshot(filepath: str = "./memory_snapshot.snapshot") -> dict:
    """
    Сохраняет текущий снапшот памяти сервера в файл.

    Args:
        filepath: Путь к файлу для сохранения снапшота.

    Returns:
        dict: Результат операции.

    Example:
        POST /admin/memory-snapshot?filepath=./snapshots/snapshot_001.snapshot
    """
    import os
    from pathlib import Path

    # Ensure directory exists
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = memory_monitor.save_snapshot(str(path))

    if success:
        file_size = os.path.getsize(path)
        return {
            "success": True,
            "filepath": str(path),
            "size_bytes": file_size,
            "message": f"Memory snapshot saved successfully"
        }
    else:
        return {
            "success": False,
            "filepath": str(path),
            "message": "Failed to save memory snapshot. Is tracemalloc enabled?"
        }


@router.get("/admin/memory-stats")
async def get_memory_stats() -> dict:
    """
    Возвращает текущую статистику использования памяти.

    Returns:
        dict: Информация о памяти процесса.
    """
    import tracemalloc
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    stats = {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "tracemalloc_enabled": tracemalloc.is_tracing(),
    }

    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        stats["tracemalloc_current_mb"] = current / (1024 * 1024)
        stats["tracemalloc_peak_mb"] = peak / (1024 * 1024)

    return stats
