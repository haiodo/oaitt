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
import threading
import time
from typing import TYPE_CHECKING, Union

import numpy as np

from src.config import TIMEOUT_ENABLED
from src.services.performance import performance_tracker

if TYPE_CHECKING:
    from src.models.schemas import TranscriptionResponse
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

# Track zombie (timed-out but still running) transcription threads
_zombie_count = 0
_zombie_lock = threading.Lock()


def get_zombie_count() -> int:
    """Returns the number of timed-out transcriptions still running in background."""
    return _zombie_count


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


def _run_transcription_with_cleanup(
    asr_model: "ASRModel",
    audio: np.ndarray,
    task: str,
    language: str | None,
    word_timestamps: bool,
    output: str,
    timed_out: threading.Event,
) -> Union["TranscriptionResponse", str]:
    """
    Wrapper that runs transcription and cleans up if the caller has timed out.

    Args:
        asr_model: ASR model instance.
        audio: Audio data.
        task: Transcription task.
        language: Language code.
        word_timestamps: Whether to include word timestamps.
        output: Output format.
        timed_out: Event that is set if the caller has already timed out.

    Returns:
        Transcription result.
    """
    global _zombie_count
    try:
        result = asr_model.transcribe(
            audio=audio,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
        )
        if timed_out.is_set():
            logger.warning(
                "Background transcription completed after caller timed out — "
                "result discarded, resources freed"
            )
        return result
    except Exception as e:
        if timed_out.is_set():
            logger.warning(
                f"Background transcription failed after caller timed out: {e}"
            )
        raise
    finally:
        if timed_out.is_set():
            with _zombie_lock:
                _zombie_count = max(0, _zombie_count - 1)
            logger.info(
                f"Zombie transcription thread finished (remaining zombies: {_zombie_count})"
            )
            # Help GC reclaim audio data sooner
            del audio


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

    Запускает транскрипцию в отдельном потоке и прерывает ожидание,
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
    """
    global _zombie_count

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

    if _zombie_count > 0:
        logger.warning(
            f"There are {_zombie_count} zombie transcription thread(s) "
            f"still running from previous timeouts"
        )

    # Event to signal the worker that the caller has timed out
    timed_out = threading.Event()

    # Run transcription in a thread with timeout
    start_time = time.perf_counter()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            _run_transcription_with_cleanup,
            asr_model=asr_model,
            audio=audio,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            output=output,
            timed_out=timed_out,
        )

        try:
            result = future.result(timeout=timeout)
            elapsed = time.perf_counter() - start_time

            # Record successful transcription for future timeout calculations
            performance_tracker.record(audio_duration_sec, elapsed)

            return result, elapsed

        except concurrent.futures.TimeoutError:
            elapsed = time.perf_counter() - start_time

            # Signal the worker that we've timed out
            timed_out.set()

            # Track this as a zombie thread
            with _zombie_lock:
                _zombie_count += 1

            logger.warning(
                f"Transcription timed out: elapsed={elapsed:.1f}s, "
                f"timeout={timeout:.1f}s, expected={expected_time:.1f}s, "
                f"audio={audio_duration_sec:.2f}s. "
                f"Background thread will continue until completion "
                f"(zombie count: {_zombie_count})"
            )

            # Shutdown executor without waiting — let the thread finish in background
            executor.shutdown(wait=False)

            raise TranscriptionTimeoutError(
                timeout=timeout,
                elapsed=elapsed,
                expected=expected_time,
            )
    finally:
        # Always shutdown executor, but don't wait for background threads
        # (they may be zombie threads from timeouts)
        executor.shutdown(wait=False)
