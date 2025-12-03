"""
OAITT — Open AI Transformer Transcriber.

Базовый абстрактный класс для ASR моделей.
Определяет интерфейс, который должны реализовать все ASR движки
(Transformers, WhisperX и т.д.).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import time
from abc import ABC, abstractmethod
from threading import Lock, Thread
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from src.config import MODEL_IDLE_TIMEOUT
from src.utils.device import clear_memory_cache

if TYPE_CHECKING:
    from src.models.schemas import TranscriptionResponse

logger = logging.getLogger(__name__)


class ASRModel(ABC):
    """
    Абстрактный базовый класс для ASR моделей.

    Определяет общий интерфейс для всех реализаций распознавания речи.
    Включает поддержку ленивой загрузки модели, мониторинга простоя
    и автоматической выгрузки неиспользуемой модели.

    Attributes:
        model: Загруженная модель (None если не загружена).
        model_lock: Lock для потокобезопасного доступа к модели.
        last_activity_time: Время последней активности (для мониторинга простоя).

    Example:
        >>> class MyASR(ASRModel):
        ...     def load_model(self):
        ...         self.model = load_whisper_model()
        ...
        ...     def transcribe(self, audio, task, language, word_timestamps, output, options=None):
        ...         result = self.model.transcribe(audio)
        ...         return TranscriptionResponse(text=result["text"])
        ...
        >>> asr = MyASR()
        >>> asr.load_model()
        >>> result = asr.transcribe(audio_data, "transcribe", "en", True, "json")
    """

    model = None
    model_lock: Lock = Lock()
    last_activity_time: float = time.time()

    def __init__(self):
        """Инициализирует ASR модель."""
        self.model = None
        self.model_lock = Lock()
        self.last_activity_time = time.time()
        self._idle_monitor_started = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Загружает модель в память.

        Должен быть реализован в подклассах. Метод должен:
        - Загрузить модель с нужными параметрами
        - Сохранить модель в self.model
        - Запустить мониторинг простоя если MODEL_IDLE_TIMEOUT > 0

        Raises:
            Exception: При ошибке загрузки модели.
        """
        pass

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union["TranscriptionResponse", str]:
        """
        Выполняет транскрипцию аудио.

        Args:
            audio: Аудиоданные в формате numpy array (16kHz, mono, float32).
            task: Задача - "transcribe" (распознавание) или "translate" (перевод на английский).
            language: Код языка (например, "en", "ru", "russian") или None для автоопределения.
            word_timestamps: Включить ли временные метки для каждого слова.
            output: Формат вывода:
                - "text": Только текст
                - "json": JSON с сегментами и метаданными
                - "vtt": WebVTT субтитры
                - "srt": SRT субтитры
                - "tsv": Tab-separated values
            options: Дополнительные опции для модели (опционально).

        Returns:
            TranscriptionResponse для JSON формата или строка для остальных форматов.

        Raises:
            Exception: При ошибке транскрипции.
        """
        pass

    def ensure_model_loaded(self) -> None:
        """
        Убеждается, что модель загружена.

        Потокобезопасно проверяет наличие модели и загружает её при необходимости.
        """
        with self.model_lock:
            if self.model is None:
                logger.info("Model not loaded, loading now...")
                self.load_model()

    def update_activity(self) -> None:
        """Обновляет время последней активности."""
        self.last_activity_time = time.time()

    def start_idle_monitor(self) -> None:
        """
        Запускает мониторинг простоя модели.

        Если MODEL_IDLE_TIMEOUT > 0, запускает фоновый поток,
        который выгрузит модель после указанного времени простоя.
        """
        if MODEL_IDLE_TIMEOUT <= 0:
            return

        if self._idle_monitor_started:
            return

        self._idle_monitor_started = True
        Thread(target=self._monitor_idleness, daemon=True).start()
        logger.info(f"Idle monitor started (timeout: {MODEL_IDLE_TIMEOUT}s)")

    def _monitor_idleness(self) -> None:
        """
        Мониторит простой модели и выгружает её при превышении таймаута.

        Выполняется в фоновом потоке. Проверяет каждые 15 секунд.
        """
        check_interval = 15  # seconds

        while True:
            time.sleep(check_interval)

            idle_time = time.time() - self.last_activity_time

            if idle_time > MODEL_IDLE_TIMEOUT:
                logger.info(
                    f"Model idle for {idle_time:.0f}s (timeout: {MODEL_IDLE_TIMEOUT}s), unloading..."
                )
                with self.model_lock:
                    self.release_model()
                self._idle_monitor_started = False
                break

    def release_model(self) -> None:
        """
        Выгружает модель из памяти.

        Освобождает ресурсы и очищает кэш памяти устройства.
        Должен вызываться с удержанным model_lock.
        """
        if self.model is not None:
            logger.info("Releasing model from memory...")

            # Allow subclasses to do custom cleanup
            self._cleanup_model()

            del self.model
            self.model = None

            clear_memory_cache()
            logger.info("Model unloaded successfully")

    def _cleanup_model(self) -> None:
        """
        Выполняет очистку специфичную для конкретной модели.

        Может быть переопределён в подклассах для дополнительной очистки.
        """
        pass

    def is_loaded(self) -> bool:
        """
        Проверяет, загружена ли модель.

        Returns:
            True если модель загружена, иначе False.
        """
        return self.model is not None

    def get_info(self) -> dict:
        """
        Возвращает информацию о модели.

        Returns:
            Словарь с информацией о модели:
            - loaded: Загружена ли модель
            - class: Имя класса модели
            - idle_timeout: Таймаут простоя
            - last_activity: Время последней активности (ISO format)
        """
        from datetime import datetime

        return {
            "loaded": self.is_loaded(),
            "class": self.__class__.__name__,
            "idle_timeout": MODEL_IDLE_TIMEOUT,
            "last_activity": datetime.fromtimestamp(self.last_activity_time).isoformat(),
        }
