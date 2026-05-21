"""
OAITT — Open AI Transformer Transcriber.

ASR реализация для GigaAM-MLX (Apple Silicon native backend).

Использует пакет `gigaam_mlx` (vendor/gigaam-mlx submodule) - порт GigaAM-v3
на MLX framework. Не требует PyTorch, работает только на Apple Silicon.

Особенности:
- CTC вариант: ~330x realtime
- RNNT вариант: ~77x realtime (выше качество)
- Веса автоматически скачиваются с HuggingFace
- Чанкование через split_audio (по тишине, до 20s)

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import os
import threading
from typing import List, Optional, Union

import numpy as np

from src.asr.base import ASRModel
from src.config import (
    GIGAAM_MLX_CHUNK_SEC,
    GIGAAM_MLX_MODEL_TYPE,
    GIGAAM_MLX_REPO_ID,
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    SAMPLE_RATE,
)
from src.models.schemas import Segment, TranscriptionResponse
from src.utils.audio import get_audio_duration, normalize_audio

logger = logging.getLogger(__name__)

# Lock-free режим по умолчанию: MLX операции lazy + thread-safe на построение графа,
# GPU исполнение сериализуется runtime'ом. Это даёт линейный throughput по клиентам
# без MODEL_WORKERS. Отключить можно через GIGAAM_MLX_LOCK_FREE=false (например, при
# нехватке unified memory - каждый параллельный клиент держит свои activations).
GIGAAM_MLX_LOCK_FREE = os.environ.get("GIGAAM_MLX_LOCK_FREE", "true").lower() == "true"

# Padding режим: дополняет аудио до GIGAAM_MLX_CHUNK_SEC секунд (default true).
# MLX JIT-компилирует kernels под каждый уникальный tensor shape - без padding это
# ведёт к высокому потреблению unified memory под warmup кеш (8 размеров = ~4 GB MPS).
# С padding один shape → один JIT-кеш → стабильная память.
# decode() получает реальный seq_len (truncates encoder output), результат корректный.
GIGAAM_MLX_PAD_TO_FIXED = os.environ.get("GIGAAM_MLX_PAD_TO_FIXED", "true").lower() == "true"


class GigaAMMLXASR(ASRModel):
    """
    ASR реализация GigaAM-MLX для Apple Silicon.

    Поддерживаемые model_type:
    - "ctc" (по умолчанию) - быстрее, ~330x realtime
    - "rnnt" - выше качество, ~77x realtime
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_type = (GIGAAM_MLX_MODEL_TYPE or "ctc").lower().strip()
        if self.model_type not in ("ctc", "rnnt"):
            logger.warning(
                f"Invalid GIGAAM_MLX_MODEL_TYPE='{self.model_type}', using 'ctc'"
            )
            self.model_type = "ctc"
        self.repo_id = GIGAAM_MLX_REPO_ID
        self.chunk_sec = GIGAAM_MLX_CHUNK_SEC

    def _resolve_local_weights(self) -> Optional[str]:
        """
        Возвращает путь к локально сконвертированным MLX весам, если они
        существуют либо могут быть автоматически сконвертированы из data/gigaam/.
        Иначе None - тогда gigaam_mlx скачает с HuggingFace.
        """
        if not MODEL_CACHE_DIR:
            return None

        mlx_dir = os.path.join(MODEL_CACHE_DIR, "gigaam_mlx", self.model_type)
        weights_path = os.path.join(mlx_dir, "weights.safetensors")
        tokenizer_path = os.path.join(mlx_dir, "tokenizer.model")
        if os.path.isfile(weights_path) and os.path.isfile(tokenizer_path):
            logger.info(f"Using locally converted MLX weights: {mlx_dir}")
            return mlx_dir

        # Не сконвертированы. Проверим, есть ли PyTorch чекпоинт от gigaam engine.
        pt_cache = os.path.join(MODEL_CACHE_DIR, "gigaam")
        pt_ckpt = os.path.join(pt_cache, f"v3_e2e_{self.model_type}.ckpt")
        if not os.path.isfile(pt_ckpt):
            return None

        logger.info(
            f"PyTorch GigaAM checkpoint found at {pt_ckpt} but MLX weights missing. "
            f"Auto-converting to {mlx_dir}..."
        )
        try:
            from scripts.convert_gigaam_to_mlx import convert_one
            return convert_one(
                self.model_type,
                gigaam_cache=pt_cache,
                output_root=os.path.join(MODEL_CACHE_DIR, "gigaam_mlx"),
            )
        except Exception as e:
            logger.warning(
                f"Auto-conversion failed ({e}); will fall back to HuggingFace download"
            )
            return None

    def load_model(self) -> None:
        """Загружает MLX модель и tokenizer."""
        try:
            # Configure HF cache directory to share with other engines
            if MODEL_CACHE_DIR and not os.environ.get("HF_HOME"):
                cache_dir = os.path.join(MODEL_CACHE_DIR, "gigaam_mlx")
                os.makedirs(cache_dir, exist_ok=True)
                os.environ["HF_HOME"] = cache_dir
                logger.info(f"GigaAM-MLX cache directory: {cache_dir}")

            # Resolve repo_id: explicit env > locally converted > HF download
            effective_repo = self.repo_id or self._resolve_local_weights()

            from gigaam_mlx import load_model as mlx_load_model

            logger.info(
                f"Loading GigaAM-MLX model: type={self.model_type}, "
                f"repo_id={effective_repo or 'HF default'}"
            )
            self.model, self.tokenizer = mlx_load_model(
                model_type=self.model_type,
                repo_id=effective_repo,
            )
            logger.info(
                f"GigaAM-MLX model '{self.model_type}' loaded successfully "
                f"(lock_free={GIGAAM_MLX_LOCK_FREE})"
            )

        except ImportError as e:
            raise ImportError(
                "gigaam_mlx package not found. Make sure vendor/gigaam-mlx is in PYTHONPATH. "
                "Use run_gigaam_mlx_asr.sh or add vendor/gigaam-mlx to your Python path."
            ) from e
        except Exception as e:
            raise Exception(f"Failed to load GigaAM-MLX model '{self.model_type}': {e}") from e

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
        """Транскрибирует аудио через MLX."""
        self.update_activity()
        self.ensure_model_loaded()

        if task == "translate":
            logger.warning("GigaAM-MLX does not support translation; doing transcription")
        if word_timestamps:
            logger.debug("GigaAM-MLX does not provide word-level timestamps")

        audio = normalize_audio(audio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        duration = get_audio_duration(audio)

        if GIGAAM_MLX_LOCK_FREE:
            segments = self._transcribe_audio(audio)
        else:
            with self.model_lock:
                segments = self._transcribe_audio(audio)

        return self._format_result(segments, duration=duration, output=output, language="ru")

    def _transcribe_audio(self, audio: np.ndarray) -> List[dict]:
        """Чанкует и транскрибирует аудио через MLX."""
        import mlx.core as mx
        from gigaam_mlx.audio import compute_mel, split_audio

        chunks = split_audio(audio, max_chunk_sec=self.chunk_sec, sr=SAMPLE_RATE)
        results: List[dict] = []

        # При padding всегда дополняем до self.chunk_sec - один JIT-кеш на все размеры.
        pad_samples = int(self.chunk_sec * SAMPLE_RATE) if GIGAAM_MLX_PAD_TO_FIXED else None

        for ch in chunks:
            chunk_audio = audio[ch["start_sample"]:ch["end_sample"]]
            real_len = len(chunk_audio)

            if pad_samples is not None and real_len < pad_samples:
                # Pad audio с нулями до фиксированного размера - mel будет фикс shape.
                chunk_audio = np.pad(chunk_audio, (0, pad_samples - real_len))

            mel = compute_mel(chunk_audio, sr=SAMPLE_RATE)
            mel_mx = mx.array(mel[np.newaxis])

            encoded, seq_len = self.model.encode(mel_mx)

            # При padding пересчитываем seq_len из реальной длины аудио.
            # Mel: hop=160, win=320, center=False -> T_mel = (real_len - 320) // 160 + 1.
            # Pre-encode: 2x Conv1d stride=2 padding=(k-1)//2=2 -> floor((T+1)/2) each.
            if pad_samples is not None and real_len < pad_samples:
                t_mel = max(0, (real_len - 320) // 160 + 1)
                # Two stride-2 convs with same padding -> ceil(T/2) twice
                t_after = (t_mel + 1) // 2
                t_after = (t_after + 1) // 2
                real_seq_len = max(1, t_after)
                seq_len = min(seq_len, real_seq_len)

            mx.eval(encoded)
            token_ids = self.model.decode(encoded, seq_len)
            text = self.tokenizer.decode(token_ids).strip()

            if text:
                results.append({
                    "text": text,
                    "start": ch["start_sec"],
                    "end": ch["end_sec"],
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
                logger.debug("Failed to delete GigaAM-MLX model", exc_info=True)
            finally:
                self.model = None
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                logger.debug("Failed to delete GigaAM-MLX tokenizer", exc_info=True)
            finally:
                self.tokenizer = None
