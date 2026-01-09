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
import tempfile
import os
import soundfile as sf

from src.asr.base import ASRModel
from src.config import (
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    WHISPER_MODEL,
    GIGAAM_REVISION,
    GIGAAM_MAX_SHORT_AUDIO_SEC,
    GIGAAM_CHUNK_SEC,
    GIGAAM_MIN_CHUNK_SEC,
    SAMPLE_RATE,
)
from src.models.schemas import (
    Segment,
    TranscriptionResponse,
    WordTimestamp,
)
from src.utils.device import get_device, is_mps_device
from src.utils.audio import get_audio_duration

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
        # Generic model support (e.g., GigaAM loaded via AutoModel / trust_remote_code)
        self.generic_model = None
        self.is_generic_model = False

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
        Загружает модель через transformers pipeline.

        Попытка загрузки в следующем порядке:
        1. Whisper-specific pipeline (WhisperForConditionalGeneration + WhisperProcessor)
        2. Fallback: Generic `AutoModel.from_pretrained(..., trust_remote_code=True)` для
           моделей, несовместимых с Whisper (например, GigaAM на HF).
        """
        device = get_device()
        self.torch_dtype = self._get_torch_dtype(device)

        logger.info(
            f"Loading Transformers model '{WHISPER_MODEL}' "
            f"on device: {device}, dtype: {self.torch_dtype}"
        )

        cache_dir = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None

        # First try loading as Whisper
        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
                pipeline,
            )

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
            self.is_generic_model = False
            logger.info("Transformers Whisper model loaded successfully")

        except Exception as e:
            # Fallback to generic AutoModel (trust_remote_code=True) for non-Whisper HF models
            logger.info(
                f"Whisper-specific loading failed or model is not a Whisper type: {e}. "
                "Attempting to load generic AutoModel with trust_remote_code=True."
            )
            try:
                from transformers import AutoModel
            except Exception as imp_err:
                raise Exception(
                    "Transformers does not appear to be available or import of AutoModel failed."
                ) from imp_err

            # Try loading AutoModel. If it fails because the model's dynamic code needs
            # extra dependencies (e.g. hydra/omegaconf/torchaudio), provide a user-friendly error
            # with actionable instructions instead of a generic traceback.
            try:
                try:
                    self.generic_model = AutoModel.from_pretrained(
                        WHISPER_MODEL,
                        revision=GIGAAM_REVISION if GIGAAM_REVISION else None,
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                    )
                except Exception as first_exc:
                    msg = str(first_exc)
                    # Detect the special transformers message listing missing packages
                    if "requires the following packages" in msg or "not found in your environment" in msg or "hydra" in msg or "omegaconf" in msg:
                        # Attempt to extract package names if present
                        missing = None
                        try:
                            import re
                            m = re.search(r"requires the following packages that were not found in your environment: (.+?)\.", msg)
                            if m:
                                missing = [p.strip() for p in m.group(1).split(",")]
                        except Exception:
                            missing = None

                        pkg_list_display = ", ".join(missing) if missing else "required packages (e.g. hydra)"
                        install_hint = f"pip install {' '.join(missing)}" if missing else "pip install hydra"
                        raise RuntimeError(
                            f"Failed to load model '{WHISPER_MODEL}' via transformers because the model's custom code requires additional packages: {pkg_list_display}. "
                            f"Install them in your environment (e.g. `{install_hint}`), or install/use the `gigaam` package or set ASR_ENGINE='gigaam'.\nOriginal error: {first_exc}"
                        ) from first_exc

                    # If not a missing-deps case, try loading without explicit revision as a fallback
                    self.generic_model = AutoModel.from_pretrained(
                        WHISPER_MODEL, trust_remote_code=True, cache_dir=cache_dir
                    )

            except RuntimeError:
                # Re-raise our friendly message (with hints about missing packages)
                raise
            except Exception as e_auto:
                # Generic failure loading AutoModel; surface helpful context
                raise Exception(
                    f"AutoModel fallback failed to load model '{WHISPER_MODEL}': {e_auto}. "
                    "If this model uses custom code, ensure all optional dependencies are installed, "
                    "or prefer using the published `gigaam` package."
                ) from e_auto

            # Attempt to place model on the chosen device
            try:
                self.generic_model.to(device)
            except Exception:
                logger.debug("Failed to move generic model to device; continuing on current device")

            self.model = self.generic_model
            self.is_generic_model = True
            logger.info("Generic transformers model loaded; will use model.transcribe() when available")

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

        Поддерживает два режима:
        - стандартный pipeline (для Whisper и совместимых моделей)
        - generic model (AutoModel / custom) — вызывается `model.transcribe(path)` / `model.transcribe_longform(path)`
        """
        self.update_activity()

        # Ensure model is loaded
        self.ensure_model_loaded()

        # If we loaded a generic model (e.g. GigaAM via AutoModel), call its transcribe API
        if self.is_generic_model:
            duration = get_audio_duration(audio)
            tmp_path = None
            # If we need to fall back from longform -> chunked transcribe, record a warning
            fallback_warning = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(tmp_path, audio, SAMPLE_RATE)

                with self.model_lock:
                    # Prefer longform when audio exceeds configured threshold and method exists
                    if duration > GIGAAM_MAX_SHORT_AUDIO_SEC and hasattr(self.model, "transcribe_longform"):
                        logger.info(f"Using model.transcribe_longform() for duration={duration:.2f}s")
                        try:
                            raw = self.model.transcribe_longform(tmp_path)
                        except Exception as long_exc:
                            # If longform fails (for example, VAD/pyannote couldn't load),
                            # attempt a safe chunked fallback using the model.transcribe() API.
                            logger.exception("model.transcribe_longform() failed; attempting chunked fallback using model.transcribe()")
                            # Record a warning so it can be surfaced in the final output
                            try:
                                fallback_warning = f"transcribe_longform failed; falling back to chunked transcribe: {long_exc}"
                            except Exception:
                                fallback_warning = "transcribe_longform failed; falling back to chunked transcribe (exception message unavailable)"
                            if hasattr(self.model, "transcribe"):
                                # Chunk size (seconds) - configured via GIGAAM_CHUNK_SEC / GIGAAM_MIN_CHUNK_SEC
                                CHUNK_SEC = GIGAAM_CHUNK_SEC
                                MIN_CHUNK_SEC = GIGAAM_MIN_CHUNK_SEC
                                chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
                                num_samples = len(audio)
                                pos = 0
                                chunks = []
                                # Iterate over audio in chunks and transcribe each piece, recursively splitting if needed
                                while pos < num_samples:
                                    end_pos = min(pos + chunk_samples, num_samples)
                                    chunk_audio = audio[pos:end_pos]
                                    start_sec = pos / SAMPLE_RATE
                                    try:
                                        subchunks = self._transcribe_chunk_with_retry(
                                            chunk_audio, start_sec, CHUNK_SEC, MIN_CHUNK_SEC
                                        )
                                    except Exception:
                                        # If even recursive splitting couldn't transcribe, propagate the error
                                        raise
                                    for sc in subchunks:
                                        text = sc.get("text") or ""
                                        bounds = sc.get("boundaries")
                                        if text:
                                            chunks.append({"text": text, "boundaries": bounds})
                                    pos = end_pos
                                # Provide the assembled chunks as the raw result (list of dicts),
                                # which will be handled by the existing list-processing path below.
                                raw = chunks
                            else:
                                # No available fallback; re-raise the original longform exception
                                raise long_exc
                    elif hasattr(self.model, "transcribe"):
                        logger.info(f"Using model.transcribe() for duration={duration:.2f}s")
                        raw = self.model.transcribe(tmp_path)
                    else:
                        raise RuntimeError("Loaded generic transformers model does not expose transcribe() API")

                # Normalize raw results into TranscriptionResponse or text
                if isinstance(raw, str):
                    if output == "text":
                        return raw
                    resp_obj = TranscriptionResponse(text=raw.strip(), language=language)
                    if fallback_warning:
                        d = resp_obj.model_dump(exclude_none=True)
                        d.setdefault("_warnings", []).append(fallback_warning)
                        return d
                    return resp_obj

                if isinstance(raw, dict) and "text" in raw:
                    if output == "text":
                        return raw.get("text", "")
                    resp_obj = TranscriptionResponse(text=raw.get("text", "").strip(), language=language)
                    if fallback_warning:
                        d = resp_obj.model_dump(exclude_none=True)
                        d.setdefault("_warnings", []).append(fallback_warning)
                        return d
                    return resp_obj

                if isinstance(raw, (list, tuple)):
                    segments = []
                    texts = []
                    for idx, utt in enumerate(raw):
                        if isinstance(utt, dict):
                            utt_text = utt.get("transcription") or utt.get("text") or ""
                            boundaries = utt.get("boundaries") or utt.get("time") or None
                            start = boundaries[0] if boundaries and len(boundaries) >= 1 else 0.0
                            end = boundaries[1] if boundaries and len(boundaries) >= 2 else start
                        else:
                            utt_text = str(utt)
                            start = 0.0
                            end = 0.0

                        utt_text = utt_text.strip()
                        texts.append(utt_text)
                        segments.append(Segment(id=idx, start=start, end=end, text=utt_text))

                    full_text = " ".join(t for t in texts if t)
                    if output == "text":
                        return full_text
                    resp = TranscriptionResponse(text=full_text.strip(), language=language, segments=segments)
                    if fallback_warning:
                        d = resp.model_dump(exclude_none=True)
                        d.setdefault("_warnings", []).append(fallback_warning)
                        return d
                    return resp

                # Fallback: stringify unknown formats
                text = str(raw)
                if output == "text":
                    return text
                resp_obj = TranscriptionResponse(text=text, language=language)
                if fallback_warning:
                    d = resp_obj.model_dump(exclude_none=True)
                    d.setdefault("_warnings", []).append(fallback_warning)
                    return d
                return resp_obj

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    def _transcribe_chunk_with_retry(self, chunk_audio: np.ndarray, start_sec: float, max_chunk_sec: float, min_chunk_sec: float):
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
                # Note: caller holds model_lock when invoking fallback, so we do not re-acquire it here
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
                        # Give up and re-raise to be handled by caller
                        raise
                    # Split into halves and recurse
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

        # Otherwise use the Whisper pipeline flow
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

                # Compute characters per second for this segment (avoid division by zero)
                seg_text = chunk.get("text", "").strip()
                duration = end_time - start_time
                chars_per_second = (
                    round(len(seg_text) / duration, 4) if duration and duration > 0 else None
                )

                segment = Segment(
                    id=idx,
                    start=start_time,
                    end=end_time,
                    text=seg_text,
                    chars_per_second=chars_per_second,
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
                    # Compute chars/sec for combined segment (avoid division by zero)
                    duration = last_end - first_start
                    seg_text = response.text
                    chars_per_second = (
                        round(len(seg_text) / duration, 4) if duration and duration > 0 else None
                    )
                    response.segments = [
                        Segment(
                            id=0,
                            start=first_start,
                            end=last_end,
                            text=seg_text,
                            words=all_words,
                            chars_per_second=chars_per_second,
                        )
                    ]
            else:
                response.segments = segments

        return response

    def _cleanup_model(self) -> None:
        """Очищает ресурсы pipeline или generic model."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if getattr(self, "generic_model", None) is not None:
            try:
                del self.generic_model
            except Exception:
                pass
            self.generic_model = None

        self.is_generic_model = False
