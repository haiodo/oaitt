"""
OAITT — Open AI Transformer Transcriber.

Сервис управления таймаутами транскрипции.
Предоставляет функции для выполнения транскрипции с адаптивными
таймаутами, основанными на исторических данных производительности.
Помогает обнаруживать зависания модели (галлюцинации, бесконечные циклы).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Union

import numpy as np

from src.config import TIMEOUT_ENABLED
from src.services.performance import performance_tracker

if TYPE_CHECKING:
    from src.models.schemas import TranscriptionResponse
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)


class TranscriptionTimeoutError(Exception):
    """
    Исключение при превышении таймаута транскрипции.

    Возникает когда транскрипция занимает слишком много времени,
    что может указывать на галлюцинации модели или проблемы с аудио.

    Attributes:
        timeout: Установленный таймаут в секундах.
        elapsed: Фактическое прошедшее время в секундах.
        expected: Ожидаемое время обработки в секундах.
    """

    def __init__(self, timeout: float, elapsed: float, expected: float):
        self.timeout = timeout
        self.elapsed = elapsed
        self.expected = expected
        super().__init__(
            f"Transcription timed out after {elapsed:.1f}s "
            f"(timeout={timeout:.1f}s, expected={expected:.1f}s)"
        )


def transcribe_with_timeout(
    asr_model: "ASRModel",
    audio: np.ndarray,
    audio_duration_sec: float,
    task: str,
    language: str | None,
    word_timestamps: bool,
    output: str,
) -> tuple[Union["TranscriptionResponse", str], float]:
    """
    Выполняет транскрипцию с адаптивным таймаутом.

    Запускает транскрипцию в отдельном потоке и прерывает её,
    если она превышает вычисленный таймаут на основе исторических
    данных производительности.

    Args:
        asr_model: Модель ASR для транскрипции.
        audio: Аудиоданные в формате numpy array.
        audio_duration_sec: Длительность аудио в секундах.
        task: Задача - "transcribe" или "translate".
        language: Код языка (например, "en", "ru") или None для автоопределения.
        word_timestamps: Включить ли временные метки слов.
        output: Формат вывода - "text", "json", "vtt", "srt", "tsv".

    Returns:
        Tuple[result, elapsed_time_sec]:
            - result: Результат транскрипции (TranscriptionResponse или str)
            - elapsed_time_sec: Время выполнения в секундах

    Raises:
        TranscriptionTimeoutError: Если транскрипция превысила таймаут.

    Example:
        >>> result, elapsed = transcribe_with_timeout(
        ...     asr_model=model,
        ...     audio=audio_data,
        ...     audio_duration_sec=30.0,
        ...     task="transcribe",
        ...     language="en",
        ...     word_timestamps=True,
        ...     output="json",
        ... )
        >>> print(f"Transcription took {elapsed:.2f}s")
    """
    if not TIMEOUT_ENABLED:
        # Timeout disabled - run directly
        start_time = time.perf_counter()
        result = asr_model.transcribe(
            audio=audio,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
        )
        elapsed = time.perf_counter() - start_time
        return result, elapsed

    # Calculate adaptive timeout
    expected_time = performance_tracker.get_expected_time(audio_duration_sec)
    timeout = performance_tracker.get_timeout(audio_duration_sec)

    logger.info(
        f"Transcription timeout: expected={expected_time:.2f}s, "
        f"timeout={timeout:.2f}s (audio={audio_duration_sec:.2f}s)"
    )

    # Run transcription in a thread with timeout
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            asr_model.transcribe,
            audio=audio,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
        )

        try:
            result = future.result(timeout=timeout)
            elapsed = time.perf_counter() - start_time

            # Record successful transcription for future timeout calculations
            performance_tracker.record(audio_duration_sec, elapsed)

            return result, elapsed

        except concurrent.futures.TimeoutError:
            elapsed = time.perf_counter() - start_time
            # Cancel the future (though it may continue running in background)
            future.cancel()

            logger.warning(
                f"Transcription timed out: elapsed={elapsed:.1f}s, "
                f"timeout={timeout:.1f}s, expected={expected_time:.1f}s, "
                f"audio={audio_duration_sec:.2f}s"
            )

            raise TranscriptionTimeoutError(
                timeout=timeout,
                elapsed=elapsed,
                expected=expected_time,
            )
