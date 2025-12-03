"""
OAITT — Open AI Transformer Transcriber.

Сервис отслеживания производительности транскрипции.
Записывает время обработки относительно длительности аудио
и вычисляет ожидаемое время обработки для новых запросов
на основе исторических данных.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from threading import Lock
from typing import Optional

from src.config import (
    TIMEOUT_MULTIPLIER,
    TIMEOUT_MIN_SECONDS,
    TIMEOUT_MAX_SECONDS,
    TIMEOUT_HISTORY_SIZE,
)

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Трекер производительности транскрипции.

    Записывает время обработки относительно длительности аудио
    и вычисляет ожидаемое время обработки для новых запросов
    на основе исторических данных. Используется для реализации
    адаптивных таймаутов.

    Attributes:
        history_size: Максимальное количество записей в истории.
        history: Список кортежей (audio_duration_sec, processing_time_sec).

    Example:
        >>> tracker = PerformanceTracker(history_size=100)
        >>> tracker.record(audio_duration_sec=30.0, processing_time_sec=3.0)
        >>> tracker.get_average_ratio()  # ~0.1 (10x realtime)
        0.1
        >>> tracker.get_timeout(audio_duration_sec=60.0)  # Expected: 6s * multiplier
        12.0
    """

    def __init__(self, history_size: int = 100):
        """
        Инициализирует трекер производительности.

        Args:
            history_size: Максимальное количество записей для хранения.
        """
        self.history_size = history_size
        # Store tuples of (audio_duration_sec, processing_time_sec)
        self.history: list[tuple[float, float]] = []
        self.lock = Lock()
        # Cached ratio (processing_time / audio_duration)
        self._avg_ratio: Optional[float] = None

    def record(self, audio_duration_sec: float, processing_time_sec: float) -> None:
        """
        Записывает результат завершённой транскрипции.

        Args:
            audio_duration_sec: Длительность аудио в секундах.
            processing_time_sec: Время обработки в секундах.
        """
        if audio_duration_sec <= 0:
            logger.warning(f"Invalid audio duration: {audio_duration_sec}")
            return

        if processing_time_sec < 0:
            logger.warning(f"Invalid processing time: {processing_time_sec}")
            return

        with self.lock:
            self.history.append((audio_duration_sec, processing_time_sec))
            # Keep only recent history
            if len(self.history) > self.history_size:
                self.history = self.history[-self.history_size:]
            # Invalidate cached ratio
            self._avg_ratio = None

            ratio = processing_time_sec / audio_duration_sec
            speed = 1.0 / ratio if ratio > 0 else float("inf")
            logger.debug(
                f"Recorded performance: audio={audio_duration_sec:.2f}s, "
                f"processing={processing_time_sec:.2f}s, "
                f"ratio={ratio:.4f}, speed={speed:.1f}x realtime"
            )

    def get_average_ratio(self) -> Optional[float]:
        """
        Возвращает среднее соотношение processing_time / audio_duration.

        Использует взвешенное среднее, где вес определяется длительностью
        аудио (более длинные записи имеют больший вес).

        Returns:
            Среднее соотношение или None, если нет данных.
        """
        with self.lock:
            if self._avg_ratio is not None:
                return self._avg_ratio

            if not self.history:
                return None

            # Calculate weighted average (give more weight to similar-length audios)
            total_audio = sum(h[0] for h in self.history)
            total_processing = sum(h[1] for h in self.history)

            if total_audio <= 0:
                return None

            self._avg_ratio = total_processing / total_audio
            return self._avg_ratio

    def get_expected_time(self, audio_duration_sec: float) -> float:
        """
        Возвращает ожидаемое время обработки для заданной длительности аудио.

        Args:
            audio_duration_sec: Длительность аудио в секундах.

        Returns:
            Ожидаемое время обработки в секундах.
            Если нет исторических данных, возвращает длительность аудио
            (консервативная оценка - realtime processing).
        """
        ratio = self.get_average_ratio()
        if ratio is None:
            # No history - use conservative estimate (realtime processing)
            return audio_duration_sec
        return audio_duration_sec * ratio

    def get_timeout(self, audio_duration_sec: float) -> float:
        """
        Вычисляет адаптивный таймаут для заданной длительности аудио.

        Таймаут = ожидаемое_время * TIMEOUT_MULTIPLIER,
        ограниченный значениями TIMEOUT_MIN и TIMEOUT_MAX.

        Args:
            audio_duration_sec: Длительность аудио в секундах.

        Returns:
            Таймаут в секундах, ограниченный min/max значениями.
        """
        expected = self.get_expected_time(audio_duration_sec)
        timeout = expected * TIMEOUT_MULTIPLIER

        # Clamp to configured bounds
        timeout = max(TIMEOUT_MIN_SECONDS, min(TIMEOUT_MAX_SECONDS, timeout))
        return timeout

    def get_stats(self) -> dict:
        """
        Возвращает статистику производительности.

        Returns:
            Словарь со статистикой:
            - samples: Количество записей в истории
            - avg_ratio: Среднее соотношение processing/audio
            - avg_speed: Средняя скорость относительно realtime
            - min_ratio: Минимальное соотношение (лучший результат)
            - max_ratio: Максимальное соотношение (худший результат)
        """
        with self.lock:
            if not self.history:
                return {
                    "samples": 0,
                    "avg_ratio": None,
                    "avg_speed": None,
                }

            ratio = self.get_average_ratio()

            # Calculate min/max ratios
            ratios = [h[1] / h[0] for h in self.history if h[0] > 0]

            return {
                "samples": len(self.history),
                "avg_ratio": round(ratio, 4) if ratio else None,
                "avg_speed": round(1.0 / ratio, 2) if ratio and ratio > 0 else None,
                "min_ratio": round(min(ratios), 4) if ratios else None,
                "max_ratio": round(max(ratios), 4) if ratios else None,
            }

    def clear(self) -> None:
        """Очищает историю производительности."""
        with self.lock:
            self.history.clear()
            self._avg_ratio = None
            logger.info("Performance history cleared")


# Global performance tracker instance
performance_tracker = PerformanceTracker(history_size=TIMEOUT_HISTORY_SIZE)
