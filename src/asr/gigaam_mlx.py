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

# Padding режим: дополняет аудио до дискретных bucket'ов (default true, шаг = bucket_sec).
# MLX JIT-компилирует kernels под каждый уникальный tensor shape - без padding каждая
# уникальная длина чанка (a-la 5.3s, 12.7s, 19.4s) даёт новый кеш и activations.
# В проде это ведёт к росту unified memory до 12+ GB при сотнях разных длин из VAD-split.
# С bucket-padding кол-во уникальных shapes = chunk_sec/bucket_sec (например 20).
# decode() получает реальный seq_len (truncates encoder output), результат корректный.
#
# Bucket size в секундах. 0 - паддить всегда до chunk_sec (макс память, 1 shape).
# 1.0 - округлять вверх до целых секунд (default, 20 shapes для chunk_sec=20).
GIGAAM_MLX_PAD_TO_FIXED = os.environ.get("GIGAAM_MLX_PAD_TO_FIXED", "true").lower() == "true"
GIGAAM_MLX_PAD_BUCKET_SEC = float(os.environ.get("GIGAAM_MLX_PAD_BUCKET_SEC", "1.0"))



def _encoder_frames(samples: int) -> int:
    """Сколько кадров энкодера даёт непаддированное аудио.

    Mel: hop=160, win=320, center=False -> T = (samples - 320) // 160 + 1.
    Pre-encode: два Conv1d со stride=2 и padding=(k-1)//2 -> ceil(T/2) дважды.
    """
    t_mel = max(0, (samples - 320) // 160 + 1)
    return max(1, ((t_mel + 1) // 2 + 1) // 2)


def _ctc_decode_batch(model, encoded, seq_lens: List[int]) -> List[List[int]]:
    """Greedy CTC по батчу: один argmax, схлопывание повторов и blank на CPU."""
    import mlx.core as mx

    labels = mx.argmax(model.head(encoded), axis=-1)
    mx.eval(labels)
    table = np.array(labels.tolist())
    blank_id = model.num_classes - 1

    hypotheses = []
    for row, length in zip(table, seq_lens):
        tokens, prev = [], blank_id
        for token in row[:length]:
            if token != blank_id and token != prev:
                tokens.append(int(token))
            prev = token
        hypotheses.append(tokens)
    return hypotheses


def _rnnt_decode_batch(model, encoded_list: list, seq_lens: List[int], max_symbols: int = 10):
    """Greedy RNNT сразу по нескольким выходам энкодера.

    Пер-чанковый цикл делает один GPU->CPU sync на кадр, чтобы прочитать argmax.
    Батч амортизирует эти синхронизации по B гипотезам: один sync на кадр на весь
    батч. Состояние LSTM продвигается маской только у строк, выдавших символ.
    """
    import mlx.core as mx

    if not seq_lens:
        return []

    batch = len(encoded_list)
    decoder, joint = model.decoder, model.joint
    blank_id, hidden_size = decoder.blank_id, decoder.pred_hidden
    max_t = max(seq_lens)
    channels = encoded_list[0].shape[1]

    padded = mx.concatenate(
        [
            enc[:, :, :max_t] if enc.shape[2] >= max_t
            else mx.concatenate([enc, mx.zeros((1, channels, max_t - enc.shape[2]))], axis=2)
            for enc in encoded_list
        ],
        axis=0,
    )

    hypotheses: List[List[int]] = [[] for _ in range(batch)]
    labels = np.zeros((batch, 1), dtype=np.int32)
    has_label = np.zeros((batch, 1, 1), dtype=np.float32)
    hidden = mx.zeros((batch, hidden_size))
    cell = mx.zeros((batch, hidden_size))

    for t in range(max_t):
        frame = mx.expand_dims(padded[:, :, t], axis=1)
        not_blank = [t < seq_lens[b] for b in range(batch)]
        symbols = 0

        while symbols < max_symbols and any(not_blank):
            emb = decoder.embed(mx.array(labels)) * mx.array(has_label)
            all_hidden, all_cell = decoder.lstm(emb, hidden, cell)
            best = np.array(mx.argmax(joint(frame, all_hidden)[:, 0, 0, :], axis=-1).tolist())

            advanced = np.zeros((batch, 1), dtype=np.float32)
            for b in range(batch):
                if not not_blank[b]:
                    continue
                if best[b] == blank_id:
                    not_blank[b] = False
                else:
                    hypotheses[b].append(int(best[b]))
                    labels[b, 0] = best[b]
                    has_label[b] = 1.0
                    advanced[b] = 1.0

            symbols += 1
            if not advanced.any():
                break

            mask = mx.array(advanced) > 0
            hidden = mx.where(mask, all_hidden[:, -1, :], hidden)
            cell = mx.where(mask, all_cell[:, -1, :], cell)

    return hypotheses


class GigaAMMLXASR(ASRModel):
    """
    ASR реализация GigaAM-MLX для Apple Silicon.

    Поддерживаемые model_type:
    - "ctc" (по умолчанию) - быстрее, ~330x realtime
    - "rnnt" - выше качество, ~77x realtime
    """

    def __init__(self, model_type: Optional[str] = None) -> None:
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_type = (model_type or GIGAAM_MLX_MODEL_TYPE or "ctc").lower().strip()
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
        from gigaam_mlx.audio import split_audio

        chunks = split_audio(audio, max_chunk_sec=self.chunk_sec, sr=SAMPLE_RATE)
        prepared = [self._prepare_chunk(audio, ch) for ch in chunks]
        prepared = [p for p in prepared if p is not None]
        if not prepared:
            return []

        texts = self.decode_batch(
            [p["mel"] for p in prepared], [p["real_frames"] for p in prepared]
        )

        return [
            {"text": text, "start": p["start_sec"], "end": p["end_sec"]}
            for p, text in zip(prepared, texts)
            if text
        ]

    def _prepare_chunk(self, audio: np.ndarray, ch: dict) -> Optional[dict]:
        """Режет чанк, паддит до бакета и считает mel."""
        import math
        from gigaam_mlx.audio import compute_mel

        chunk_audio = audio[ch["start_sample"]:ch["end_sample"]]
        real_len = len(chunk_audio)
        if real_len < 320:
            return None

        max_samples = int(self.chunk_sec * SAMPLE_RATE)
        if GIGAAM_MLX_PAD_TO_FIXED:
            bucket_samples = (
                max_samples if GIGAAM_MLX_PAD_BUCKET_SEC <= 0
                else int(GIGAAM_MLX_PAD_BUCKET_SEC * SAMPLE_RATE)
            )
            target = min(max_samples, int(math.ceil(real_len / bucket_samples)) * bucket_samples)
            if real_len < target:
                chunk_audio = np.pad(chunk_audio, (0, target - real_len))

        return {
            "mel": compute_mel(chunk_audio, sr=SAMPLE_RATE),
            "real_frames": _encoder_frames(real_len),
            "start_sec": ch["start_sec"],
            "end_sec": ch["end_sec"],
        }

    def decode_batch(self, mels: List[np.ndarray], real_frames: List[int]) -> List[str]:
        """Прогоняет чанки через энкодер и декодирует.

        Энкодер требует одинаковой формы, поэтому чанки группируются по длине mel.
        Декодер RNNT - нет: он паддит выходы энкодера до общего числа кадров сам,
        и декодировать их надо одним батчем, иначе теряется весь смысл (один
        GPU->CPU sync на кадр амортизируется по числу гипотез, а не по группе).
        """
        import mlx.core as mx

        if not mels:
            return []

        groups: dict = {}
        for index, mel in enumerate(mels):
            groups.setdefault(mel.shape, []).append(index)

        encoded_by_index: dict = {}
        lens_by_index: dict = {}
        for indices in groups.values():
            encoded, seq_len = self.model.encode(mx.array(np.stack([mels[i] for i in indices])))
            for position, index in enumerate(indices):
                encoded_by_index[index] = encoded[position : position + 1]
                lens_by_index[index] = min(seq_len, real_frames[index])
        mx.eval(list(encoded_by_index.values()))

        order = sorted(encoded_by_index)
        encoded_list = [encoded_by_index[i] for i in order]
        lens = [lens_by_index[i] for i in order]

        if self.model_type == "rnnt":
            hypotheses = _rnnt_decode_batch(self.model, encoded_list, lens)
        else:
            hypotheses = [
                _ctc_decode_batch(self.model, enc, [n])[0]
                for enc, n in zip(encoded_list, lens)
            ]
        return [self.tokenizer.decode(ids).strip() for ids in hypotheses]

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
