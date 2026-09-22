"""
OAITT — Open AI Transformer Transcriber.

ASR реализация на пакете `onnx-asr` (ONNX Runtime, CPU). Без PyTorch и MLX,
работает на любой платформе - x86_64/arm64, Linux/macOS/Windows.

Модели (см. docs/benchmarks.md):
- gigaam-v3-e2e-rnnt (default) - пунктуация и капитализация, WER 6.85% на Golos, ~41x realtime
- gigaam-v3-rnnt - без пунктуации, WER 2.18%, ~55x realtime
- nemo-parakeet-tdt-0.6b-v3 - 25 языков; int8-квантизация у апстрима сломана (мусор на
  английском), поэтому только fp32, WER 3.42%

Токенизатор - не char-level: piece'ы переменной длины ('Про', 'вер', 'я'), граница слова -
ведущий пробел в piece'е, как у Parakeet-MLX (см. parakeet_mlx.py::_tokens_to_words).

Whole-file инференс обрезает результат (модель обучена на коротких отрезках), поэтому
длинное аудио режется на куски через split_audio_smart, как у остальных движков.

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import math
from typing import List, Optional, Union

import numpy as np

from src.asr.base import ASRModel
from src.config import (
    MODEL_IDLE_TIMEOUT,
    ONNX_ASR_CHUNK_SEC,
    ONNX_ASR_MODEL,
    ONNX_ASR_MODEL_DIR,
    ONNX_ASR_QUANTIZATION,
    SAMPLE_RATE,
)
from src.models.schemas import Segment, TranscriptionResponse, WordTimestamp
from src.utils.audio import get_audio_duration, normalize_audio
from src.utils.chunking import split_audio_smart

logger = logging.getLogger(__name__)


def _group_words(
    tokens: List[str], timestamps: List[float], logprobs: Optional[List[float]]
) -> List[WordTimestamp]:
    """Группирует piece'ы токенизатора в слова: ведущий пробел в piece'е - граница слова.

    Пунктуация приходит отдельным piece'ом без пробела и приклеивается к предыдущему
    слову (' Жги' + '.'). probability - exp(mean(logprob)) по piece'ам слова.

    ponytail: onnx-asr отдаёт только onset каждого piece'а, не его длительность, поэтому
    end слова = onset последнего piece'а - для однослогового слова end==start. Точная
    длительность потребовала бы либо доп. запроса к модели, либо эвристики по соседнему
    onset'у - не делаем, пока это не понадобится потребителю API.
    """
    words: List[WordTimestamp] = []
    chars: List[str] = []
    lps: List[float] = []
    start = end = 0.0

    def flush():
        text = "".join(chars).strip()
        if not text:
            return
        prob = round(float(math.exp(sum(lps) / len(lps))), 4) if lps else None
        words.append(WordTimestamp(word=text, start=start, end=end, probability=prob))

    for i, (piece, ts) in enumerate(zip(tokens, timestamps)):
        if not piece:
            continue
        if piece.startswith(" ") and chars:
            flush()
            chars, lps = [], []
        if not chars:
            start = ts
        chars.append(piece)
        end = ts
        if logprobs is not None:
            lps.append(logprobs[i])
    flush()
    return words


class OnnxASR(ASRModel):
    """
    ASR реализация на onnx-asr (ONNX Runtime, CPU).

    Имя модели - из реестра onnx-asr: "gigaam-v3-e2e-rnnt" (default), "gigaam-v3-rnnt",
    "nemo-parakeet-tdt-0.6b-v3" и т.д.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self.model = None
        self.model_name = (model or ONNX_ASR_MODEL or "gigaam-v3-e2e-rnnt").strip()

        quant = (ONNX_ASR_QUANTIZATION or "int8").strip().lower()
        self.quantization = None if quant == "fp32" else quant
        if self.model_name.startswith("nemo-parakeet") and self.quantization == "int8":
            logger.warning(
                f"int8 quantization is broken for '{self.model_name}' upstream "
                "(produces English garbage); falling back to fp32"
            )
            self.quantization = None

    def load_model(self) -> None:
        """Загружает onnx-asr модель (веса скачиваются с HuggingFace при первом запуске)."""
        try:
            import onnx_asr

            logger.info(
                f"Loading onnx-asr model: {self.model_name} "
                f"(quantization={self.quantization or 'fp32'})"
            )
            self.model = onnx_asr.load_model(
                self.model_name,
                path=ONNX_ASR_MODEL_DIR or None,
                quantization=self.quantization,
                providers=["CPUExecutionProvider"],
            )
            logger.info("onnx-asr model loaded successfully")
        except ImportError as e:
            raise ImportError(
                'onnx-asr package not found. Install it with: pip install "onnx-asr[cpu,hub]"'
            ) from e
        except Exception as e:
            raise Exception(f"Failed to load onnx-asr model '{self.model_name}': {e}") from e

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
        """Транскрибирует аудио через onnx-asr (onnxruntime потокобезопасен, без лока)."""
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning("onnx-asr does not support translation; doing transcription")
        # word_timestamps: with_timestamps() почти бесплатен, слова собираются всегда.

        audio = normalize_audio(audio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        duration = get_audio_duration(audio)

        raw_segments = self._run_transcribe(audio)
        language_hint = "ru" if self.model_name.startswith("gigaam") else None
        return self._format_result(
            raw_segments, duration=duration, output=output, language=language_hint
        )

    def _run_transcribe(self, audio: np.ndarray) -> List[dict]:
        """Режет аудио на куски и распознаёт каждый, сдвигая временные метки на офсет куска."""
        segments = []
        for chunk, start, end in split_audio_smart(audio, ONNX_ASR_CHUNK_SEC):
            if len(chunk) == 0:
                continue
            result = self.model.with_timestamps().recognize(chunk, sample_rate=SAMPLE_RATE)
            text = (result.text or "").strip()
            if not text:
                continue
            words = None
            if result.tokens and result.timestamps:
                shifted = [t + start for t in result.timestamps]
                words = _group_words(result.tokens, shifted, result.logprobs) or None
            segments.append({"text": text, "start": start, "end": end, "words": words})
        return segments

    def _format_result(
        self,
        raw_segments: List[dict],
        duration: float,
        output: str,
        language: Optional[str] = None,
    ) -> Union[TranscriptionResponse, str]:
        """Собирает TranscriptionResponse из кусков."""
        texts: List[str] = []
        segments: List[Segment] = []

        for idx, seg in enumerate(raw_segments):
            text = seg["text"]
            texts.append(text)
            start, end = seg["start"], seg["end"]
            dur = end - start
            segments.append(
                Segment(
                    id=idx,
                    start=start,
                    end=end,
                    text=text,
                    words=seg.get("words"),
                    chars_per_second=round(len(text) / dur, 4) if dur > 0 else None,
                )
            )

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
        """Освобождение ресурсов модели."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                logger.debug("Failed to delete onnx-asr model", exc_info=True)
            finally:
                self.model = None
