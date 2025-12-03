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
from src.services.performance import performance_tracker

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

    return {
        "status": "healthy",
        "model_loaded": _asr_model is not None and _asr_model.is_loaded(),
        "engine": ASR_ENGINE,
        "timeout_enabled": TIMEOUT_ENABLED,
        "performance": perf_stats,
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
