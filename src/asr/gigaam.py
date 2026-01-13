"""
OAITT — Open AI Transformer Transcriber.

ASR реализация для моделей семейства GigaAM.

Интегрирует пакет `gigaam` и предоставляет интерфейс совместимый
с базовым `ASRModel`, который использует остальная часть сервиса.

Особенности реализации:
- Загружает модель через `gigaam.load_model(model_name)`
- Для коротких аудио (по умолчанию <= 25s) использует `.transcribe(path)`
- Для длинных аудио делит аудио на куски (GIGAAM_CHUNK_SEC) и транскрибирует каждый кусок через `.transcribe(path)`
- Результат конвертируется в `TranscriptionResponse` (сегменты при длинном аудио)
- Неявно использует HF/TORCH кэш, настроенный через переменные окружения

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import os
import tempfile
from typing import Optional, Union, List

import numpy as np
import soundfile as sf

from src.asr.base import ASRModel
from src.config import (
    GIGAAM_MODEL,
    GIGAAM_REVISION,
    GIGAAM_MAX_SHORT_AUDIO_SEC,
    GIGAAM_CHUNK_SEC,
    GIGAAM_MIN_CHUNK_SEC,
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    SAMPLE_RATE,
)
from src.models.schemas import Segment, TranscriptionResponse, WordTimestamp
from src.utils.audio import get_audio_duration, normalize_audio

logger = logging.getLogger(__name__)


class GigaAMASR(ASRModel):
    """
    ASR реализация для GigaAM.

    Использует пакет `gigaam` для загрузки и выполнения инференса.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.model_name = GIGAAM_MODEL

    def load_model(self) -> None:
        """
        Загружает модель GigaAM.

        Raises:
            Exception: Если пакет `gigaam` не установлен или модель не удалось загрузить.
        """
        # Prefer the `gigaam` package loader if it is available.
        try:
            import gigaam  # Local import to avoid hard dependency at module import time
            logger.info("Detected 'gigaam' package; loading model via gigaam.load_model()")
            # Some versions accept download_root/cache options; rely on HF env vars instead.
            self.model = gigaam.load_model(self.model_name)
            logger.info("GigaAM model loaded successfully via 'gigaam' package")
        except Exception as pkg_exc:
            # Fallback to Hugging Face transformers AutoModel (trust_remote_code=True)
            logger.info(
                "gigaam package not available or failed to load; attempting to load via Hugging Face transformers AutoModel"
            )
            try:
                from transformers import AutoModel  # Local import to avoid hard dependency

                logger.info(
                    f"Loading GigaAM from HF: repo={GIGAAM_MODEL}, revision={GIGAAM_REVISION}"
                )
                cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None

                # Pass revision only if provided
                self.model = AutoModel.from_pretrained(
                    GIGAAM_MODEL,
                    revision=GIGAAM_REVISION if GIGAAM_REVISION else None,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                )

                logger.info("GigaAM model loaded successfully via transformers AutoModel")
            except Exception as hf_exc:
                # Both approaches failed - provide actionable error
                raise Exception(
                    "Failed to load GigaAM model: neither 'gigaam' package nor transformers AutoModel succeeded. "
                    "Install the GigaAM package or ensure the HF repo 'ai-sage/GigaAM-v3' and its dependencies are accessible."
                ) from hf_exc

        # Move model to the selected device (if applicable) and start idle monitor
        try:
            from src.utils.device import get_device

            device = get_device()
            moved = False

            # First, try moving the model itself (works for transformers PreTrainedModel and similar)
            try:
                if hasattr(self.model, "to"):
                    self.model.to(device)
                    logger.info(f"Moved GigaAM model to device: {device}")
                    moved = True
            except Exception as e_move:
                logger.debug(f"Moving model to device failed: {e_move}")

            # Fallback: try moving an inner torch module if exposed (e.g., `.model`)
            if not moved:
                try:
                    inner = getattr(self.model, "model", None)
                    if inner is not None and hasattr(inner, "to"):
                        inner.to(device)
                        logger.info(f"Moved inner GigaAM model to device: {device}")
                        moved = True
                except Exception as e_inner:
                    logger.debug(f"Moving inner model to device failed: {e_inner}")

            if not moved:
                logger.debug("Could not move GigaAM model to device; continuing on current device")

        except Exception as e:
            logger.debug(f"Device selection or model move failed: {e}")

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
        Выполняет транскрипцию аудио используя GigaAM.

        Подход:
        - Сохраняет numpy->wav во временный файл (GigaAM API ожидает путь к файлу)
        - Вызывает `.transcribe(path)` для коротких аудио
        - Для длинных аудио (более GIGAAM_MAX_SHORT_AUDIO_SEC) делит аудио на куски (GIGAAM_CHUNK_SEC) и последовательно транскрибирует куски через `.transcribe(path)`
        - Форматирует результат в `TranscriptionResponse` или строку в зависимости от `output`

        Args:
            audio: numpy array (16kHz, mono, float32)
            task: \"transcribe\" или \"translate\" (GigaAM не поддерживает перевод — игнорируется)
            language: код языка (необязательно)
            word_timestamps: требуется ли уровень слов (если GigaAM вернул слова)
            output: \"text\" или \"json\" (мы возвращаем TranscriptionResponse для json)

        Returns:
            TranscriptionResponse для JSON или строку для text.
        """
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning("GigaAM does not support translation; performing transcription instead")

        # Normalize audio and compute duration
        audio = normalize_audio(audio)
        duration = get_audio_duration(audio)

        tmp_path = None
        try:
            # Choose inference method
            with self.model_lock:
                # For audio longer than the "short" threshold, perform chunked transcription using model.transcribe()
                if duration > GIGAAM_MAX_SHORT_AUDIO_SEC:
                    if not hasattr(self.model, "transcribe"):
                        raise RuntimeError("Loaded GigaAM model does not expose transcribe() API required for chunked transcription")
                    logger.info(
                        f"Audio duration {duration:.2f}s exceeds {GIGAAM_MAX_SHORT_AUDIO_SEC}s; performing chunked transcribe using model.transcribe()"
                    )
                    CHUNK_SEC = GIGAAM_CHUNK_SEC
                    MIN_CHUNK_SEC = GIGAAM_MIN_CHUNK_SEC
                    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
                    num_samples = len(audio)
                    pos = 0
                    chunks = []
                    # Iterate over audio in chunks and transcribe each piece, splitting further if needed
                    while pos < num_samples:
                        end_pos = min(pos + chunk_samples, num_samples)
                        chunk_audio = audio[pos:end_pos]
                        start_sec = pos / SAMPLE_RATE
                        try:
                            subchunks = self._transcribe_chunk_with_retry(
                                chunk_audio, start_sec, CHUNK_SEC, MIN_CHUNK_SEC
                            )
                        except Exception:
                            # If we cannot transcribe even after recursive splitting, propagate
                            raise
                        for sc in subchunks:
                            text = sc.get("text") or ""
                            bounds = sc.get("boundaries")
                            if text:
                                chunks.append({"text": text, "boundaries": bounds})
                        pos = end_pos
                    raw_result = chunks
                else:
                    # Write to a temporary WAV file for GigaAM to consume
                    # Use delete=False because some OS won't allow reopening NamedTemporaryFile on Windows
                    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    sf.write(tmp_path, audio, SAMPLE_RATE)
                    logger.debug(f"Saved temporary audio to {tmp_path} for GigaAM inference")
                    logger.info(f"Using GigaAM.transcribe for duration={duration:.2f}s")
                    raw_result = self.model.transcribe(tmp_path)

        finally:
            # Clean up temp file if it was created (chunked path uses its own temp files)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.debug("Failed to remove temporary audio file", exc_info=True)

        # Format and return result
        return self._format_result(
            raw_result, duration=duration, output=output, language=language, want_words=word_timestamps
        )

    def _transcribe_chunk_with_retry(
        self,
        chunk_audio: np.ndarray,
        start_sec: float,
        max_chunk_sec: float,
        min_chunk_sec: float,
    ):
        """
        Attempt to transcribe `chunk_audio`. If `model.transcribe()` raises a ValueError
        indicating the chunk is too long (or asks to use longform), split the chunk into halves
        and retry recursively until pieces are small enough or the minimum chunk size is reached.

        Returns a list of dicts: [{"text": "...", "boundaries": (start, end)}, ...]
        """
        results: list[dict] = []
        duration = len(chunk_audio) / SAMPLE_RATE
        if duration <= 0:
            return results

        # If chunk is within allowed size, try to transcribe directly
        if duration <= max_chunk_sec:
            path = None
            try:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(path, chunk_audio, SAMPLE_RATE)
                # Caller holds model_lock when invoking this method, so do not re-acquire it
                raw_chunk = self.model.transcribe(path)

                # Normalize raw_chunk into text
                if isinstance(raw_chunk, str):
                    text = raw_chunk.strip()
                elif isinstance(raw_chunk, dict):
                    text = (raw_chunk.get("text") or raw_chunk.get("transcription") or "").strip()
                elif isinstance(raw_chunk, (list, tuple)):
                    pieces = []
                    for u in raw_chunk:
                        if isinstance(u, dict):
                            pieces.append(u.get("transcription") or u.get("text") or "")
                        else:
                            pieces.append(str(u))
                    text = " ".join(p for p in pieces if p).strip()
                else:
                    text = str(raw_chunk).strip()

                if text:
                    results.append({"text": text, "boundaries": (start_sec, start_sec + duration)})
                return results

            except ValueError as ve:
                msg = str(ve)
                # If the model complains it's too long, split further and retry
                if ("Too long wav file" in msg) or ("transcribe_longform" in msg) or ("Too long" in msg):
                    if duration <= min_chunk_sec:
                        # Give up and re-raise
                        raise
                    mid = len(chunk_audio) // 2
                    first = chunk_audio[:mid]
                    second = chunk_audio[mid:]
                    mid_sec = start_sec + mid / SAMPLE_RATE
                    results.extend(self._transcribe_chunk_with_retry(first, start_sec, max_chunk_sec, min_chunk_sec))
                    results.extend(self._transcribe_chunk_with_retry(second, mid_sec, max_chunk_sec, min_chunk_sec))
                    return results
                # If it's a different ValueError, propagate
                raise
            finally:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

        # If chunk is larger than allowed max, split into halves and recurse
        mid = len(chunk_audio) // 2
        first = chunk_audio[:mid]
        second = chunk_audio[mid:]
        mid_sec = start_sec + mid / SAMPLE_RATE
        results.extend(self._transcribe_chunk_with_retry(first, start_sec, max_chunk_sec, min_chunk_sec))
        results.extend(self._transcribe_chunk_with_retry(second, mid_sec, max_chunk_sec, min_chunk_sec))
        return results

    def _format_result(
        self,
        raw_result,
        duration: float,
        output: str,
        language: Optional[str],
        want_words: bool = False,
    ) -> Union[TranscriptionResponse, str]:
        """
        Преобразует «сырые» результаты GigaAM в TranscriptionResponse или строку.

        Поддерживает несколько возможных форматов возвращаемых данных:
        - str: простой текст (CTC/RNNT краткий результат)
        - dict: возможный словарь с ключами 'text' / 'transcription' / 'segments'
        - list: список для longform, где каждый элемент — utterance (dict) с 'transcription' и 'boundaries'
        """
        # Short string result
        if isinstance(raw_result, str):
            text = raw_result.strip()
            if output == "text":
                return text
            resp = TranscriptionResponse(text=text, language=language)
            # chars/sec calculated at higher level (routes/openai) but we can populate it
            if duration and duration > 0:
                resp.chars_per_second = round(len(text) / duration, 4)
            return resp

        # Dictionary-like result
        if isinstance(raw_result, dict):
            # Typical keys: 'text' or 'transcription', optionally 'segments'
            text = raw_result.get("text") or raw_result.get("transcription") or ""
            if output == "text":
                return text

            response = TranscriptionResponse(text=text.strip(), language=language)
            if duration and duration > 0:
                response.chars_per_second = round(len(response.text) / duration, 4)

            # If segments provided by GigaAM, convert them
            segments_raw = raw_result.get("segments")
            if segments_raw:
                segments = []
                for idx, seg in enumerate(segments_raw):
                    start = seg.get("start", 0)
                    end = seg.get("end", start)
                    seg_text = seg.get("text", "").strip()
                    chars_per_second = None
                    dur = end - start
                    if dur and dur > 0:
                        chars_per_second = round(len(seg_text) / dur, 4)
                    words = None
                    # If per-word timestamps exist, map them
                    if want_words and "words" in seg:
                        words = []
                        for w in seg.get("words", []):
                            words.append(
                                WordTimestamp(
                                    word=w.get("word", ""),
                                    start=w.get("start", 0),
                                    end=w.get("end", 0),
                                    probability=w.get("probability", None),
                                )
                            )
                    segments.append(
                        Segment(
                            id=idx,
                            start=start,
                            end=end,
                            text=seg_text,
                            words=words,
                            chars_per_second=chars_per_second,
                        )
                    )
                response.segments = segments

            return response

        # List-like result (longform utterances)
        if isinstance(raw_result, (list, tuple)):
            segments: List[Segment] = []
            texts: List[str] = []

            for idx, utt in enumerate(raw_result):
                if isinstance(utt, dict):
                    utt_text = utt.get("transcription") or utt.get("text") or ""
                    boundaries = utt.get("boundaries") or utt.get("time") or None
                    start = boundaries[0] if boundaries and len(boundaries) >= 1 else 0.0
                    end = boundaries[1] if boundaries and len(boundaries) >= 2 else start
                else:
                    # Fallback - utterance is plain string
                    utt_text = str(utt)
                    start = 0.0
                    end = 0.0

                utt_text = utt_text.strip()
                texts.append(utt_text)

                duration_seg = end - start
                chars_per_second = round(len(utt_text) / duration_seg, 4) if duration_seg and duration_seg > 0 else None

                segments.append(
                    Segment(
                        id=idx,
                        start=start,
                        end=end,
                        text=utt_text,
                        chars_per_second=chars_per_second,
                    )
                )

            full_text = " ".join(t for t in texts if t)
            if output == "text":
                return full_text

            response = TranscriptionResponse(text=full_text.strip(), language=language, segments=segments)
            if duration and duration > 0:
                response.chars_per_second = round(len(response.text) / duration, 4)
            return response

        # Unknown format - return safe string or empty response
        logger.warning("GigaAM returned unsupported result format; returning raw string representation")
        text = str(raw_result)
        if output == "text":
            return text
        return TranscriptionResponse(text=text, language=language)

    def _cleanup_model(self) -> None:
        """Очистка ресурсов модели GigaAM."""
        if self.model is not None:
            try:
                # Some models may have a `.close()` or `.to(None)` method, but we don't assume it.
                del self.model
            except Exception:
                logger.debug("Failed to delete GigaAM model object cleanly", exc_info=True)
            finally:
                self.model = None
