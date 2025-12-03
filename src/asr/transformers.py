"""
OAITT — Open AI Transformer Transcriber.

ASR реализация на Hugging Face Transformers.
Использует pipeline из transformers для распознавания речи
с моделями Whisper. Поддерживает CUDA, MPS (Apple Silicon) и CPU.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import time
from threading import Thread
from typing import Optional, Union

import numpy as np
import torch

from src.asr.base import ASRModel
from src.config import MODEL_CACHE_DIR, MODEL_IDLE_TIMEOUT, WHISPER_MODEL
from src.models.schemas import (
    Segment,
    TranscriptionResponse,
    WordTimestamp,
)
from src.utils.device import get_device, is_mps_device

logger = logging.getLogger(__name__)


class TransformersASR(ASRModel):
    """
    ASR реализация на Hugging Face Transformers.

    Использует transformers pipeline с моделями Whisper.
    Поддерживает различные устройства (CUDA, MPS, CPU) и автоматически
    выбирает оптимальный тип данных для каждого.

    Attributes:
        pipeline: Transformers pipeline для распознавания речи.
        torch_dtype: Тип данных PyTorch (float16 для MPS, bfloat16 для других).

    Example:
        >>> asr = TransformersASR()
        >>> asr.load_model()
        >>> result = asr.transcribe(
        ...     audio=audio_data,
        ...     task="transcribe",
        ...     language="en",
        ...     word_timestamps=True,
        ...     output="json",
        ... )
        >>> print(result.text)
    """

    def __init__(self):
        """Инициализирует TransformersASR."""
        super().__init__()
        self.pipeline = None
        self.torch_dtype = None  # Will be set in load_model based on device

    def _get_torch_dtype(self, device: torch.device) -> torch.dtype:
        """
        Определяет оптимальный тип данных для устройства.

        Args:
            device: Устройство для инференса.

        Returns:
            torch.dtype: Оптимальный тип данных.
            - float16 для MPS (ограниченная поддержка bfloat16)
            - bfloat16 для CUDA и CPU
        """
        if is_mps_device(device):
            # MPS has limited bfloat16 support, use float16 instead
            return torch.float16
        return torch.bfloat16

    def load_model(self) -> None:
        """
        Загружает модель Whisper через transformers pipeline.

        Загружает модель, процессор и создаёт pipeline для
        автоматического распознавания речи.

        Raises:
            Exception: При ошибке загрузки модели.
        """
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            pipeline,
        )

        device = get_device()
        self.torch_dtype = self._get_torch_dtype(device)

        logger.info(
            f"Loading Transformers Whisper model '{WHISPER_MODEL}' "
            f"on device: {device}, dtype: {self.torch_dtype}"
        )

        cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None

        # Load model with optimizations
        whisper = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir,
        )

        # Load processor (tokenizer + feature extractor)
        processor = WhisperProcessor.from_pretrained(
            WHISPER_MODEL,
            cache_dir=cache_dir,
        )

        # Create pipeline
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=device,
        )

        self.model = self.pipeline
        logger.info("Transformers Whisper model loaded successfully")

        # Start idle monitor if configured
        if MODEL_IDLE_TIMEOUT > 0:
            self.start_idle_monitor()

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        """
        Выполняет транскрипцию аудио.

        Args:
            audio: Аудиоданные (16kHz, mono, float32).
            task: "transcribe" или "translate".
            language: Код языка или None для автоопределения.
            word_timestamps: Включить временные метки слов.
            output: Формат вывода ("text", "json", "vtt", "srt", "tsv").
            options: Дополнительные опции (не используются).

        Returns:
            TranscriptionResponse для JSON или строка для остальных форматов.
        """
        self.update_activity()

        # Ensure model is loaded
        self.ensure_model_loaded()

        # Prepare generation kwargs
        generate_kwargs = {"max_new_tokens": 256}

        if language:
            generate_kwargs["language"] = language

        if task == "translate":
            generate_kwargs["task"] = "translate"

        # Determine if we need word-level timestamps
        return_ts = word_timestamps and output == "json"

        # Run inference
        result = self.pipeline(
            audio,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if return_ts else True,
        )

        logger.info(
            f"Transcription completed: {len(result.get('text', ''))} characters"
        )

        return self._format_result(result, language, output, return_ts)

    def _format_result(
        self,
        result: dict,
        language: Optional[str],
        output: str,
        return_timestamps: bool,
    ) -> Union[TranscriptionResponse, str]:
        """
        Форматирует результат транскрипции.

        Args:
            result: Сырой результат от pipeline.
            language: Код языка.
            output: Формат вывода.
            return_timestamps: Включены ли временные метки слов.

        Returns:
            Отформатированный результат.
        """
        # For text output, just return the text
        if output == "text":
            return result.get("text", "")

        # Build response object
        response = TranscriptionResponse(
            text=result.get("text", "").strip(),
            language=language,
        )

        # Process chunks if available
        if "chunks" in result and result["chunks"]:
            segments = []
            all_words = []

            for idx, chunk in enumerate(result["chunks"]):
                timestamp = chunk.get("timestamp", (0, 0))
                start_time = timestamp[0] if timestamp[0] is not None else 0
                end_time = timestamp[1] if timestamp[1] is not None else start_time

                segment = Segment(
                    id=idx,
                    start=start_time,
                    end=end_time,
                    text=chunk.get("text", "").strip(),
                )

                # Collect words if word timestamps are enabled
                if return_timestamps:
                    word = WordTimestamp(
                        word=chunk.get("text", "").strip(),
                        start=start_time,
                        end=end_time,
                    )
                    all_words.append(word)

                segments.append(segment)

            # For word timestamps, create a single segment with all words
            if return_timestamps and all_words:
                if segments:
                    first_start = segments[0].start
                    last_end = segments[-1].end
                    response.segments = [
                        Segment(
                            id=0,
                            start=first_start,
                            end=last_end,
                            text=response.text,
                            words=all_words,
                        )
                    ]
            else:
                response.segments = segments

        return response

    def _cleanup_model(self) -> None:
        """Очищает ресурсы pipeline."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
