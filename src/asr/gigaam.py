"""
OAITT — Open AI Transformer Transcriber.

ASR реализация для моделей семейства GigaAM.

Интегрирует пакет `gigaam` (из vendor/gigaam submodule) и предоставляет
интерфейс совместимый с базовым `ASRModel`.

Особенности реализации:
- Загружает модель через `gigaam.load_model(model_name)`
- Для коротких аудио (по умолчанию <= 25s) использует `.transcribe(path)`
- Для длинных аудио использует `.transcribe_longform(path)` если доступен HF_TOKEN,
  иначе делит аудио на куски и транскрибирует каждый кусок через `.transcribe(path)`
- Результат конвертируется в `TranscriptionResponse` (сегменты при длинном аудио)

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import gc
import logging
import os
import tempfile
from typing import Optional, Union, List

import numpy as np
import soundfile as sf
import torch

from src.asr.base import ASRModel
from src.config import (
    GIGAAM_MODEL,
    GIGAAM_MAX_SHORT_AUDIO_SEC,
    GIGAAM_CHUNK_SEC,
    GIGAAM_MIN_CHUNK_SEC,
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    SAMPLE_RATE,
)

# Enable longform transcription via VAD (requires HF_TOKEN and pyannote access)
# Set GIGAAM_USE_LONGFORM=true to enable, otherwise chunked transcription is used
GIGAAM_USE_LONGFORM = os.environ.get("GIGAAM_USE_LONGFORM", "false").lower() == "true"
from src.models.schemas import Segment, TranscriptionResponse, WordTimestamp
from src.utils.audio import get_audio_duration, normalize_audio
from src.utils.device import clear_memory_cache

logger = logging.getLogger(__name__)

# Memory cleanup configuration
# Cleanup every N transcriptions to prevent gradual memory growth
GIGAAM_CLEANUP_EVERY_N = int(os.environ.get("GIGAAM_CLEANUP_EVERY_N", "100"))
# Force cleanup if memory grew by this amount (MB) since last cleanup
GIGAAM_CLEANUP_MEMORY_THRESHOLD_MB = int(os.environ.get("GIGAAM_CLEANUP_MEMORY_THRESHOLD_MB", "2048"))

# Default model name for GigaAM
DEFAULT_GIGAAM_MODEL = "v3_e2e_rnnt"

# Maximum recursion depth for chunk splitting to prevent unbounded memory usage
_MAX_CHUNK_SPLIT_DEPTH = 10


class GigaAMASR(ASRModel):
    """
    ASR реализация для GigaAM.

    Использует пакет `gigaam` из vendor/gigaam для загрузки и выполнения инференса.

    Поддерживаемые модели:
    - v3_e2e_rnnt (по умолчанию) - лучшее качество с пунктуацией
    - v3_e2e_ctc - end-to-end с пунктуацией
    - v3_rnnt, v3_ctc - без пунктуации
    - v2_rnnt, v2_ctc - предыдущая версия
    - v1_rnnt, v1_ctc - первая версия
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        # Use GIGAAM_MODEL from config, default to v3_e2e_rnnt
        self.model_name = GIGAAM_MODEL if GIGAAM_MODEL else DEFAULT_GIGAAM_MODEL
        # Check if it's an HF path like "ai-sage/GigaAM-v3", convert to gigaam model name
        if "/" in self.model_name:
            # It's an HF path, extract model type or use default
            self.model_name = DEFAULT_GIGAAM_MODEL
            logger.info(f"Detected HF path in GIGAAM_MODEL, using default: {self.model_name}")

        # Memory tracking for smart cleanup
        self._transcription_count = 0
        self._last_cleanup_memory = 0
        self._last_cleanup_count = 0

    def load_model(self) -> None:
        """
        Загружает модель GigaAM через пакет gigaam.

        Raises:
            Exception: Если пакет `gigaam` не установлен или модель не удалось загрузить.
        """
        try:
            import gigaam
            logger.info(f"Loading GigaAM model: {self.model_name}")

            # Determine device
            from src.utils.device import get_device
            device = get_device()
            device_str = str(device)

            # Configure download directory - use data/gigaam subdirectory
            if MODEL_CACHE_DIR:
                download_root = os.path.join(MODEL_CACHE_DIR, "gigaam")
                # Ensure directory exists
                os.makedirs(download_root, exist_ok=True)
                logger.info(f"GigaAM cache directory: {download_root}")
            else:
                download_root = None

            # Load model with appropriate settings
            self.model = gigaam.load_model(
                model_name=self.model_name,
                fp16_encoder=True if "cuda" in device_str else False,
                use_flash=False,  # Disable flash attention for compatibility
                device=device,
                download_root=download_root,
            )

            logger.info(f"GigaAM model '{self.model_name}' loaded successfully on device: {device}")

        except ImportError as e:
            raise ImportError(
                "GigaAM package not found. Make sure vendor/gigaam is in PYTHONPATH. "
                "Use run_gigaam_asr.sh or add vendor/gigaam to your Python path manually."
            ) from e
        except Exception as e:
            raise Exception(f"Failed to load GigaAM model '{self.model_name}': {e}") from e

        # Start idle monitor if configured
        if MODEL_IDLE_TIMEOUT > 0:
            self.start_idle_monitor()

    def _maybe_cleanup_memory(self) -> None:
        """
        Smart memory cleanup - only when needed.
        
        Cleanup triggers:
        1. Every GIGAAM_CLEANUP_EVERY_N transcriptions
        2. When memory grew by GIGAAM_CLEANUP_MEMORY_THRESHOLD_MB since last cleanup
        
        This avoids expensive cleanup after every transcription while preventing
        gradual memory growth. See: https://github.com/pytorch/pytorch/issues/154329
        """
        self._transcription_count += 1
        
        # Check if cleanup needed
        cleanup_needed = False
        reason = ""
        
        # Check count threshold
        if self._transcription_count - self._last_cleanup_count >= GIGAAM_CLEANUP_EVERY_N:
            cleanup_needed = True
            reason = f"count threshold ({self._transcription_count} transcriptions)"
        
        # Check memory threshold (MPS only)
        if not cleanup_needed and torch.backends.mps.is_available():
            try:
                current_mem = torch.mps.driver_allocated_memory() / (1024 * 1024)  # MB
                if self._last_cleanup_memory > 0:
                    mem_growth = current_mem - self._last_cleanup_memory
                    if mem_growth >= GIGAAM_CLEANUP_MEMORY_THRESHOLD_MB:
                        cleanup_needed = True
                        reason = f"memory growth ({mem_growth:.0f} MB)"
            except Exception:
                pass  # Ignore errors in memory tracking
        
        if cleanup_needed:
            logger.debug(f"Memory cleanup triggered: {reason}")
            gc.collect()
            clear_memory_cache()
            
            # Update tracking
            self._last_cleanup_count = self._transcription_count
            if torch.backends.mps.is_available():
                try:
                    self._last_cleanup_memory = torch.mps.driver_allocated_memory() / (1024 * 1024)
                except Exception:
                    pass

    @torch.inference_mode()
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

        Args:
            audio: numpy array (16kHz, mono, float32)
            task: "transcribe" или "translate" (GigaAM не поддерживает перевод)
            language: код языка - только для поля в ответе. Модель язык на вход
                не принимает: v3 обучена на русском, multilingual_* на 70+ языках
                и определяет язык сама
            word_timestamps: требуется ли уровень слов (GigaAM не поддерживает)
            output: "text" или "json"
            options: дополнительные опции (не используются)

        Returns:
            TranscriptionResponse для JSON или строку для text.
        """
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning("GigaAM does not support translation; performing transcription instead")

        if word_timestamps:
            logger.debug("GigaAM does not provide word-level timestamps")

        # Normalize audio and compute duration
        audio = normalize_audio(audio)
        duration = get_audio_duration(audio)

        tmp_path = None
        try:
            with self.model_lock:
                # Check if we should use longform or chunked transcription
                if duration > GIGAAM_MAX_SHORT_AUDIO_SEC:
                    # Use longform only if explicitly enabled and HF_TOKEN is set
                    hf_token = os.environ.get("HF_TOKEN")
                    use_longform = (
                        GIGAAM_USE_LONGFORM
                        and hf_token
                        and hasattr(self.model, "transcribe_longform")
                    )

                    if use_longform:
                        logger.info(
                            f"Audio duration {duration:.2f}s exceeds {GIGAAM_MAX_SHORT_AUDIO_SEC}s; "
                            "using transcribe_longform (VAD-based)"
                        )
                        # Write to temp file
                        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                        os.close(fd)
                        sf.write(tmp_path, audio, SAMPLE_RATE)

                        try:
                            raw_result = self.model.transcribe_longform(tmp_path)
                        except Exception as e:
                            logger.warning(
                                f"transcribe_longform failed: {e}; falling back to chunked"
                            )
                            raw_result = self._transcribe_chunked(audio)
                    else:
                        # Default: chunked transcription (more reliable)
                        logger.info(
                            f"Audio duration {duration:.2f}s exceeds {GIGAAM_MAX_SHORT_AUDIO_SEC}s; "
                            "using chunked transcription"
                        )
                        raw_result = self._transcribe_chunked(audio)
                else:
                    # Short audio - use direct tensor transcription (no temp file needed)
                    logger.info(f"Using direct tensor transcription for duration={duration:.2f}s")
                    raw_result = self._transcribe_audio_tensor(audio)

        finally:
            # Clean up temp file (only used for transcribe_longform)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.debug("Failed to remove temporary audio file", exc_info=True)

        # Format and return result
        formatted_result = self._format_result(
            raw_result, duration=duration, output=output, language=language or "ru"
        )
        
        # Clear raw_result to free any tensor references it might hold
        if 'raw_result' in locals():
            del raw_result
        
        # Smart memory cleanup - only when needed (count or memory threshold)
        # Prevents gradual memory growth while avoiding expensive cleanup every request
        self._maybe_cleanup_memory()
        
        return formatted_result

    def _transcribe_chunked(self, audio: np.ndarray) -> List[dict]:
        """
        Транскрибирует длинное аудио по частям.

        Args:
            audio: numpy array аудио данных

        Returns:
            Список словарей с транскрипцией и границами
        """
        results = []
        chunk_samples = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
        min_chunk_samples = int(GIGAAM_MIN_CHUNK_SEC * SAMPLE_RATE)
        num_samples = len(audio)

        # Build chunk boundaries first in order to avoid creating a tiny final chunk
        # (for example when audio length is slightly larger than chunk size).
        chunks = []
        pos = 0
        while pos < num_samples:
            end_pos = min(pos + chunk_samples, num_samples)
            remaining = num_samples - pos
            # If the remaining part is smaller than the minimum chunk, merge it into
            # the previous chunk to avoid very short chunks that may break feature extraction.
            if remaining < min_chunk_samples and chunks:
                prev_start, _ = chunks[-1]
                # Extend previous chunk to include the remainder
                chunks[-1] = (prev_start, num_samples)
                logger.debug(
                    "Merging tiny final remainder (%d samples, %.3fs) "
                    "into previous chunk starting at %.2fs",
                    remaining,
                    remaining / SAMPLE_RATE,
                    prev_start / SAMPLE_RATE,
                )
                break
            chunks.append((pos, end_pos))
            pos = end_pos

        for start, end in chunks:
            chunk_audio = audio[start:end]
            start_sec = start / SAMPLE_RATE

            try:
                subchunks = self._transcribe_chunk_with_retry(
                    chunk_audio, start_sec, GIGAAM_CHUNK_SEC, GIGAAM_MIN_CHUNK_SEC
                )
            except Exception as e:
                logger.error(f"Failed to transcribe chunk at {start_sec:.2f}s: {e}")
                raise

            for sc in subchunks:
                text = sc.get("text") or sc.get("transcription") or ""
                bounds = sc.get("boundaries")
                if text:
                    results.append({"transcription": text, "boundaries": bounds})

            # Clear memory after each chunk to prevent accumulation
            clear_memory_cache()

        return results

    def _transcribe_audio_tensor(self, audio: np.ndarray) -> str:
        """
        Транскрибирует аудио напрямую из numpy array без сохранения во временный файл.

        Использует внутренние методы GigaAM модели для обработки тензора напрямую,
        минуя загрузку из файла через ffmpeg.

        Args:
            audio: numpy array аудио данных (16kHz, mono, float32)

        Returns:
            Строка с транскрипцией
        """
        # WORKAROUND for MPS memory leak with variable tensor sizes
        # Issue: https://github.com/pytorch/pytorch/issues/154329
        # Solution: Pad to fixed size to prevent mps_copy_ leak
        original_len = len(audio)
        max_samples = int(GIGAAM_MAX_SHORT_AUDIO_SEC * SAMPLE_RATE)
        
        if len(audio) < max_samples:
            # Pad to fixed size - prevents MPS variable-size tensor leak
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Convert numpy array to torch tensor
        # GigaAM expects float tensor normalized to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        wav_tensor = torch.from_numpy(audio)

        # Get model device and dtype
        device = self.model._device
        dtype = self.model._dtype

        # Prepare tensor like model.prepare_wav() does, but from memory
        wav = wav_tensor.to(device).to(dtype).unsqueeze(0)
        length = torch.full([1], original_len, device=device)  # Use ORIGINAL length, not padded

        encoded = None
        encoded_len = None
        result = None
        try:
            # Run forward pass - inference_mode already set at transcribe() level
            encoded, encoded_len = self.model.forward(wav, length)
            result = self.model.decoding.decode(self.model.head, encoded, encoded_len)[0]
            # gigaam >= 0.2 отдаёт (text, token_ids, token_frames) на сэмпл,
            # раньше была просто строка.
            if isinstance(result, tuple):
                result = result[0]
        finally:
            # Clear intermediate tensors to prevent memory accumulation
            if encoded_len is not None:
                del encoded_len
            if encoded is not None:
                del encoded
            del wav, length, wav_tensor
            
            # Clear CUDA/MPS cache to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                try:
                    torch.mps.synchronize()
                except AttributeError:
                    pass

        return result

    def _transcribe_chunk_with_retry(
        self,
        chunk_audio: np.ndarray,
        start_sec: float,
        max_chunk_sec: float,
        min_chunk_sec: float,
        _depth: int = 0,
    ) -> List[dict]:
        """
        Транскрибирует кусок аудио с автоматическим разделением при ошибке.

        Args:
            chunk_audio: numpy array аудио куска
            start_sec: начальное время куска в секундах
            max_chunk_sec: максимальная длина куска
            min_chunk_sec: минимальная длина куска перед отказом

        Returns:
            Список словарей с транскрипцией и границами
        """
        results = []
        duration = len(chunk_audio) / SAMPLE_RATE

        if _depth > _MAX_CHUNK_SPLIT_DEPTH:
            logger.error(
                f"Maximum chunk split depth ({_MAX_CHUNK_SPLIT_DEPTH}) exceeded "
                f"at {start_sec:.2f}s, duration={duration:.2f}s — skipping chunk"
            )
            return results

        if duration <= 0:
            return results

        # If chunk is within allowed size, try to transcribe directly
        if duration <= max_chunk_sec:
            try:
                # Use direct tensor transcription (no temp file needed)
                raw_chunk = self._transcribe_audio_tensor(chunk_audio)

                # Normalize result to text
                if isinstance(raw_chunk, str):
                    text = raw_chunk.strip()
                elif isinstance(raw_chunk, dict):
                    text = (raw_chunk.get("text") or raw_chunk.get("transcription") or "").strip()
                else:
                    text = str(raw_chunk).strip()

                if text:
                    results.append({
                        "text": text,
                        "boundaries": (start_sec, start_sec + duration)
                    })
                return results

            except (ValueError, RuntimeError) as ve:
                msg = str(ve)
                # If the model complains it's too long, split further
                if "Too long" in msg or "longform" in msg.lower():
                    if duration <= min_chunk_sec:
                        raise
                    # Split in half and retry
                    mid = len(chunk_audio) // 2
                    first = chunk_audio[:mid]
                    second = chunk_audio[mid:]
                    mid_sec = start_sec + mid / SAMPLE_RATE
                    results.extend(self._transcribe_chunk_with_retry(
                        first, start_sec, max_chunk_sec, min_chunk_sec, _depth=_depth + 1
                    ))
                    results.extend(self._transcribe_chunk_with_retry(
                        second, mid_sec, max_chunk_sec, min_chunk_sec, _depth=_depth + 1
                    ))
                    return results
                raise

        # If chunk is larger than allowed max, split into halves
        mid = len(chunk_audio) // 2
        first = chunk_audio[:mid]
        second = chunk_audio[mid:]
        mid_sec = start_sec + mid / SAMPLE_RATE
        results.extend(self._transcribe_chunk_with_retry(
            first, start_sec, max_chunk_sec, min_chunk_sec, _depth=_depth + 1
        ))
        results.extend(self._transcribe_chunk_with_retry(
            second, mid_sec, max_chunk_sec, min_chunk_sec, _depth=_depth + 1
        ))
        return results

    def _format_result(
        self,
        raw_result,
        duration: float,
        output: str,
        language: str = "ru",
    ) -> Union[TranscriptionResponse, str]:
        """
        Преобразует результаты GigaAM в TranscriptionResponse или строку.

        Args:
            raw_result: результат от GigaAM (str, dict, или list)
            duration: длительность аудио в секундах
            output: "text" или "json"
            language: код языка

        Returns:
            TranscriptionResponse или строка
        """
        # gigaam >= 0.2 отдаёт TranscriptionResult / LongformTranscriptionResult
        # вместо строки и списка dict - разворачиваем в привычные типы.
        if hasattr(raw_result, "segments"):
            raw_result = [
                {
                    "transcription": getattr(s, "text", ""),
                    "boundaries": getattr(s, "boundaries", None),
                }
                for s in raw_result.segments
            ]
        elif hasattr(raw_result, "text"):
            raw_result = raw_result.text

        # Simple string result (short audio)
        if isinstance(raw_result, str):
            text = raw_result.strip()
            if output == "text":
                return text
            resp = TranscriptionResponse(text=text, language=language)
            if duration and duration > 0:
                resp.chars_per_second = round(len(text) / duration, 4)
            return resp

        # Dictionary result
        if isinstance(raw_result, dict):
            text = raw_result.get("text") or raw_result.get("transcription") or ""
            text = text.strip()
            if output == "text":
                return text

            response = TranscriptionResponse(text=text, language=language)
            if duration and duration > 0:
                response.chars_per_second = round(len(text) / duration, 4)
            return response

        # List result (longform or chunked)
        if isinstance(raw_result, (list, tuple)):
            segments: List[Segment] = []
            texts: List[str] = []

            for idx, utt in enumerate(raw_result):
                if isinstance(utt, dict):
                    utt_text = utt.get("transcription") or utt.get("text") or ""
                    boundaries = utt.get("boundaries")
                    start = boundaries[0] if boundaries and len(boundaries) >= 1 else 0.0
                    end = boundaries[1] if boundaries and len(boundaries) >= 2 else start
                else:
                    utt_text = str(utt)
                    start = 0.0
                    end = 0.0

                utt_text = utt_text.strip()
                if utt_text:
                    texts.append(utt_text)

                    duration_seg = end - start
                    chars_per_second = (
                        round(len(utt_text) / duration_seg, 4)
                        if duration_seg and duration_seg > 0
                        else None
                    )

                    segments.append(
                        Segment(
                            id=idx,
                            start=start,
                            end=end,
                            text=utt_text,
                            chars_per_second=chars_per_second,
                        )
                    )

            full_text = " ".join(texts)
            if output == "text":
                return full_text

            response = TranscriptionResponse(
                text=full_text.strip(),
                language=language,
                segments=segments if segments else None,
            )
            if duration and duration > 0:
                response.chars_per_second = round(len(response.text) / duration, 4)
            return response

        # Unknown format
        logger.warning("GigaAM returned unsupported result format; returning raw string")
        text = str(raw_result)
        if output == "text":
            return text
        return TranscriptionResponse(text=text, language=language)

    def _cleanup_model(self) -> None:
        """Очистка ресурсов модели GigaAM."""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                logger.debug("Failed to delete GigaAM model object", exc_info=True)
            finally:
                self.model = None
