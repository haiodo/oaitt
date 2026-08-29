"""
OAITT — Open AI Transformer Transcriber.

ASR реализация для GigaAM-Multilingual на MLX (Apple Silicon).

Использует пакет `gigaam_multilingual_mlx` (PyPI) - порт GigaAM-Multilingual
(600M энкодер, charwise CTC) на MLX. Не требует PyTorch.

Особенности:
- ~210-230x realtime на M4 Max
- 70+ языков; лучший WER на русском, казахском, киргизском, узбекском
- Пунктуацию не проставляет (charwise CTC)
- Word-level таймстемпы из CTC-выравнивания
- Веса скачиваются с HuggingFace (ai-babai/gigaam-multilingual-mlx)

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np

from src.asr.base import ASRModel
from src.config import MODEL_IDLE_TIMEOUT, SAMPLE_RATE
from src.models.schemas import Segment, TranscriptionResponse
from src.utils.audio import get_audio_duration, normalize_audio

logger = logging.getLogger(__name__)

# Вариант весов: int8 (default) или fp16. На M4 Max скорость одинаковая
# (~210x), int8 занимает вдвое меньше памяти - 666 MB против 1116 MB.
GIGAAM_ML_MLX_VARIANT = os.environ.get("GIGAAM_ML_MLX_VARIANT", "int8").lower().strip()

# Чанкование: пакет режет аудио фиксированными окнами с перекрытием,
# половина перекрытия отбрасывается с каждой стороны стыка.
GIGAAM_ML_MLX_CHUNK_SEC = float(os.environ.get("GIGAAM_ML_MLX_CHUNK_SEC", "20.0"))
GIGAAM_ML_MLX_OVERLAP_SEC = float(os.environ.get("GIGAAM_ML_MLX_OVERLAP_SEC", "2.0"))

# Локальный путь к весам вместо загрузки с HuggingFace.
GIGAAM_ML_MLX_MODEL_DIR = os.environ.get("GIGAAM_ML_MLX_MODEL_DIR", "").strip()

# MLX операции lazy и thread-safe на построение графа, GPU-исполнение
# сериализует runtime - см. комментарий в gigaam_mlx.py.
GIGAAM_ML_MLX_LOCK_FREE = (
    os.environ.get("GIGAAM_ML_MLX_LOCK_FREE", "true").lower() == "true"
)


class GigaAMMultilingualMLXASR(ASRModel):
    """ASR реализация GigaAM-Multilingual MLX для Apple Silicon."""

    def __init__(self, variant: Optional[str] = None) -> None:
        super().__init__()
        self.model = None
        self.variant = (variant or GIGAAM_ML_MLX_VARIANT)
        if self.variant not in ("int8", "fp16"):
            logger.warning(
                f"Invalid GIGAAM_ML_MLX_VARIANT='{self.variant}', using 'int8'"
            )
            self.variant = "int8"
        self.chunk_sec = GIGAAM_ML_MLX_CHUNK_SEC
        self.overlap_sec = GIGAAM_ML_MLX_OVERLAP_SEC

    def load_model(self) -> None:
        """Загружает модель GigaAM-Multilingual MLX."""
        if self.model is not None:
            return

        try:
            from gigaam_multilingual_mlx import load_model as load_mlx_model
        except ImportError as e:
            raise ImportError(
                "gigaam_multilingual_mlx package not found. "
                "Install it with: pip install gigaam-multilingual-mlx"
            ) from e

        model_dir = GIGAAM_ML_MLX_MODEL_DIR or None
        logger.info(
            f"Loading GigaAM-Multilingual MLX (variant={self.variant}, "
            f"dir={model_dir or 'HuggingFace'})"
        )

        try:
            self.model = load_mlx_model(model_dir, variant=self.variant)
        except Exception as e:
            raise Exception(
                f"Failed to load GigaAM-Multilingual MLX '{self.variant}': {e}"
            ) from e

        logger.info(f"GigaAM-Multilingual MLX loaded (variant={self.variant})")

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
        """Транскрибирует аудио через GigaAM-Multilingual MLX."""
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning(
                "GigaAM-Multilingual does not support translation; doing transcription"
            )

        audio = normalize_audio(audio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        duration = get_audio_duration(audio)

        if GIGAAM_ML_MLX_LOCK_FREE:
            segments = self._transcribe_audio(audio)
        else:
            with self.model_lock:
                segments = self._transcribe_audio(audio)

        return self._format_result(
            segments, duration=duration, output=output, language=language or "ru"
        )

    def _transcribe_audio(self, audio: np.ndarray) -> List[dict]:
        """Чанкует и транскрибирует аудио, отдаёт сегменты с таймингами."""
        import mlx.core as mx
        from gigaam_multilingual_mlx.audio import fixed_chunks

        results: List[dict] = []
        chunk_ranges = list(fixed_chunks(audio, self.chunk_sec, self.overlap_sec))
        last = len(chunk_ranges) - 1

        for index, (start, end, samples) in enumerate(chunk_ranges):
            logits, lengths = self.model(
                mx.array(samples)[None, :], mx.array([len(samples)])
            )
            mx.eval(logits, lengths)
            # greedy_decode отдаёт dict: text, token_ids, token_frames.
            decoded = self.model.greedy_decode(logits, lengths)[0]
            text = (decoded.get("text") or "").strip()
            if not text:
                continue

            # Отбрасываем половину перекрытия с каждой стороны стыка, чтобы
            # текст на границе чанков не дублировался.
            start_sec = start / SAMPLE_RATE
            end_sec = end / SAMPLE_RATE
            results.append({
                "text": text,
                "start": start_sec if index == 0 else start_sec + self.overlap_sec / 2,
                "end": end_sec if index == last else end_sec - self.overlap_sec / 2,
            })

        return results

    def _format_result(
        self,
        raw_segments: List[dict],
        duration: float,
        output: str,
        language: str = "ru",
    ) -> Union[TranscriptionResponse, str]:
        """Форматирует результат в TranscriptionResponse или строку."""
        texts: List[str] = []
        segments: List[Segment] = []

        for idx, seg in enumerate(raw_segments):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            dur = end - start
            cps = round(len(text) / dur, 4) if dur > 0 else None
            segments.append(Segment(
                id=idx, start=start, end=end, text=text, chars_per_second=cps
            ))

        full_text = " ".join(texts).strip()
        if output == "text":
            return full_text

        response = TranscriptionResponse(
            text=full_text,
            language=language,
            segments=segments if segments else None,
        )
        if duration and duration > 0 and full_text:
            response.chars_per_second = round(len(full_text) / duration, 4)
        return response

    def _cleanup_model(self) -> None:
        """Освобождение ресурсов MLX модели."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                logger.debug("Failed to delete model", exc_info=True)
            finally:
                self.model = None
