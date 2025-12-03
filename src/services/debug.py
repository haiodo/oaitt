"""
OAITT — Open AI Transformer Transcriber.

Сервис отладочного логирования.
Сохраняет аудиофайлы и результаты транскрипции для отладки
и анализа качества распознавания.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Union

import numpy as np
import soundfile as sf

from src.config import DEBUG_LOG_DIR, SAMPLE_RATE

if TYPE_CHECKING:
    from src.models.schemas import TranscriptionResponse

logger = logging.getLogger(__name__)


def save_debug_log(
    audio_data: np.ndarray,
    result: Union["TranscriptionResponse", str],
    filename: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Сохраняет аудио и результат транскрипции для отладки.

    Создаёт два файла:
    - {timestamp}_{filename}.wav - Аудиофайл (16kHz mono)
    - {timestamp}_{filename}.json - Результат транскрипции

    Args:
        audio_data: Аудиоданные в формате numpy array (16kHz float32).
        result: Результат транскрипции (TranscriptionResponse или строка).
        filename: Опциональное имя исходного файла для включения в имя.

    Returns:
        Tuple[str | None, str | None]: Пути к сохранённым файлам (audio_path, result_path)
        или (None, None) если логирование отключено.

    Note:
        Функция ничего не делает, если DEBUG_LOG_DIR не установлен.
        Ошибки при сохранении логируются, но не прерывают выполнение.
    """
    if not DEBUG_LOG_DIR:
        return None, None

    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = os.path.splitext(filename or "audio")[0]
        # Sanitize filename - remove potentially problematic characters
        base_name = "".join(c for c in base_name if c.isalnum() or c in "._-")[:50]
        prefix = f"{timestamp}_{base_name}"

        # Save audio as WAV
        audio_path = os.path.join(DEBUG_LOG_DIR, f"{prefix}.wav")
        sf.write(audio_path, audio_data, SAMPLE_RATE)
        logger.debug(f"Debug audio saved: {audio_path}")

        # Save transcription result as JSON
        result_path = os.path.join(DEBUG_LOG_DIR, f"{prefix}.json")

        if isinstance(result, str):
            result_data = {"text": result}
        else:
            result_data = result.model_dump(exclude_none=True)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Debug result saved: {result_path}")

        logger.info(f"Debug log saved: {prefix}")
        return audio_path, result_path

    except Exception as e:
        logger.warning(f"Failed to save debug log: {e}")
        return None, None


def cleanup_old_logs(max_age_hours: int = 24, max_files: int = 1000) -> int:
    """
    Удаляет старые файлы отладки.

    Args:
        max_age_hours: Максимальный возраст файлов в часах.
        max_files: Максимальное количество файлов для сохранения.

    Returns:
        Количество удалённых файлов.

    Note:
        Функция ничего не делает, если DEBUG_LOG_DIR не установлен.
    """
    if not DEBUG_LOG_DIR:
        return 0

    import glob
    from pathlib import Path

    deleted_count = 0

    try:
        # Get all debug files
        pattern = os.path.join(DEBUG_LOG_DIR, "*.*")
        files = glob.glob(pattern)

        if not files:
            return 0

        # Sort by modification time (oldest first)
        files_with_time = [(f, os.path.getmtime(f)) for f in files]
        files_with_time.sort(key=lambda x: x[1])

        # Delete files exceeding max_files limit
        if len(files_with_time) > max_files:
            files_to_delete = files_with_time[: len(files_with_time) - max_files]
            for file_path, _ in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

        # Delete files older than max_age_hours
        import time

        cutoff_time = time.time() - (max_age_hours * 3600)
        for file_path, mtime in files_with_time:
            if mtime < cutoff_time and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old debug files")

    except Exception as e:
        logger.warning(f"Failed to cleanup debug logs: {e}")

    return deleted_count


def get_debug_log_stats() -> dict:
    """
    Возвращает статистику директории отладочных логов.

    Returns:
        Словарь со статистикой:
        - enabled: Включено ли логирование
        - path: Путь к директории
        - file_count: Количество файлов
        - total_size_mb: Общий размер в мегабайтах
    """
    if not DEBUG_LOG_DIR:
        return {"enabled": False}

    try:
        import glob

        pattern = os.path.join(DEBUG_LOG_DIR, "*.*")
        files = glob.glob(pattern)

        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))

        return {
            "enabled": True,
            "path": DEBUG_LOG_DIR,
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    except Exception as e:
        logger.warning(f"Failed to get debug log stats: {e}")
        return {
            "enabled": True,
            "path": DEBUG_LOG_DIR,
            "error": str(e),
        }
