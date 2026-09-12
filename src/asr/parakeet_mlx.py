"""
OAITT — Open AI Transformer Transcriber.

ASR реализация для Parakeet-TDT-v3 (NVIDIA, 25 европейских языков) через пакет
`parakeet_mlx` с PyPI. Не требует PyTorch, работает только на Apple Silicon.

Особенности:
- ~72x realtime на M4 Max
- WER на Golos (общий набор) 3.99% - лучше GigaAM RNNT (6.69%)
- Word-level timestamps и пер-токенная confidence из коробки
- Длинное аудио - окнами с перекрытием (120s/15s по умолчанию)

Важно: все MLX-вызовы идут через один выделенный поток. В mlx 0.32.x eval графа,
задевающего CPU-stream (как у Parakeet), из не-главного потока падает с
"There is no Stream(cpu, N) in current thread"; переход на один поток инференса
обходит это ценой сериализации запросов - GPU всё равно исполняет их последовательно.

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import os
import queue
import tempfile
import threading
from typing import List, Optional, Union

import numpy as np

from src.asr.base import ASRModel
from src.config import (
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    PARAKEET_CHUNK_SEC,
    PARAKEET_OVERLAP_SEC,
    PARAKEET_REPO_ID,
    PARAKEET_REPO_ID_INT8,
)
from src.models.schemas import Segment, TranscriptionResponse, WordTimestamp
from src.utils.audio import get_audio_duration, normalize_audio

logger = logging.getLogger(__name__)


def _tokens_to_words(tokens: list) -> List[WordTimestamp]:
    """Группирует piece'ы токенизатора в слова.

    Словарь Parakeet - BPE: границей слова служит ведущий пробел в piece'е
    (' Про' + 'ве' + 'ря' + 'ем'). Для совместимости с SentencePiece-словарями
    границей считается и `▁`. Слово: start первого piece, end последнего,
    probability = min по piece'ам - консервативная оценка для фильтра качества.
    """
    words: List[WordTimestamp] = []
    current: Optional[dict] = None

    def flush():
        nonlocal current
        if current is not None and current["word"]:
            words.append(
                WordTimestamp(
                    word=current["word"],
                    start=current["start"],
                    end=current["end"],
                    probability=current["prob"],
                )
            )
        current = None

    for tok in tokens:
        raw = tok.text or ""
        piece = raw.strip()
        if not piece:
            continue
        if raw.startswith((" ", "▁")) or current is None:
            flush()
            current = {
                "word": piece,
                "start": tok.start,
                "end": tok.end,
                "prob": tok.confidence,
            }
        else:
            current["word"] += piece
            current["end"] = tok.end
            current["prob"] = min(current["prob"], tok.confidence)
    flush()
    return words


class ParakeetMLXASR(ASRModel):
    """
    ASR реализация Parakeet-TDT-v3 для Apple Silicon.

    Поддерживаемые variant:
    - "fp16" (по умолчанию) - mlx-community/parakeet-tdt-0.6b-v3
    - "int8" - sonic-speech/parakeet-tdt-0.6b-v3-int8, +30% скорости
    """

    def __init__(self, variant: Optional[str] = None) -> None:
        super().__init__()
        self.model = None
        self.variant = (variant or os.environ.get("PARAKEET_VARIANT") or "fp16").lower().strip()
        if self.variant not in ("fp16", "int8"):
            logger.warning(f"Invalid parakeet variant '{self.variant}', using 'fp16'")
            self.variant = "fp16"
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._worker_ready = threading.Event()

    @property
    def repo_id(self) -> str:
        return PARAKEET_REPO_ID_INT8 if self.variant == "int8" else PARAKEET_REPO_ID

    # ------------------------------------------------------------------ worker

    def _start_worker(self) -> None:
        """Поднимает поток инференса один раз; загрузка модели идёт в нём же."""
        if self._worker_ready.is_set():
            return

        with self.model_lock:
            if self._worker_ready.is_set():
                return
            threading.Thread(target=self._worker_loop, daemon=True, name="parakeet-mlx").start()
            self._worker_ready.wait()

    def _worker_loop(self) -> None:
        self._worker_ready.set()
        while True:
            audio, duration, done, holder = self._jobs.get()
            try:
                with self.model_lock:
                    if self.model is None:
                        self.load_model()
                    model = self.model
                holder["result"] = self._run_transcribe(model, audio, duration)
            except Exception as e:  # noqa: BLE001 - наверх уходит исключение запроса
                logger.exception("Parakeet inference failed")
                holder["error"] = e
            finally:
                done.set()

    # -------------------------------------------------------------------- api

    def ensure_model_loaded(self) -> None:
        """Модель грузится в потоке инференса, а не в вызывающем.

        Загрузка в главном потоке + eval графа в рабочем ломает mlx 0.32.x
        (thread-local CPU streams), поэтому весь MLX-контекст живёт в одном
        выделенном потоке.
        """
        self._start_worker()

    def load_model(self) -> None:
        """Загружает Parakeet модель. Вызывается в потоке инференса."""
        try:
            if MODEL_CACHE_DIR and not os.environ.get("HF_HOME"):
                cache_dir = os.path.join(MODEL_CACHE_DIR, "parakeet")
                os.makedirs(cache_dir, exist_ok=True)
                os.environ["HF_HOME"] = cache_dir
                logger.info(f"Parakeet cache directory: {cache_dir}")

            from parakeet_mlx import from_pretrained

            logger.info(f"Loading Parakeet model: repo={self.repo_id}, variant={self.variant}")
            if self.variant == "int8":
                self.model = self._load_int8()
            else:
                self.model = from_pretrained(self.repo_id)
            logger.info("Parakeet model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "parakeet_mlx package not found. Install it with: pip install parakeet-mlx"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to load Parakeet model '{self.repo_id}': {e}") from e

        if MODEL_IDLE_TIMEOUT > 0:
            self.start_idle_monitor()

    def _load_int8(self):
        """Квантованный чекпойнт sonic-speech: энкодер 8 бит, group_size 64.

        from_pretrained такие чекпойнты не читает (в дереве модели нет
        scales/biases), поэтому собираем по схеме из карточки модели:
        nn.quantize(model.encoder, bits=8, group_size=64) и загрузка весов.
        Декодер и joint остаются в исходной точности.
        """
        import json
        from pathlib import Path

        import mlx.nn as nn
        from huggingface_hub import hf_hub_download
        from parakeet_mlx.utils import from_config

        try:
            weight = hf_hub_download(self.repo_id, "model.safetensors")
            config_path = Path(weight).parent / "config.json"
        except Exception:
            weight = str(Path(self.repo_id) / "model.safetensors")
            config_path = Path(self.repo_id) / "config.json"

        config = json.load(open(config_path))
        model = from_config(config)
        nn.quantize(model.encoder, bits=8, group_size=64)
        model.load_weights(weight)
        model.eval()
        return model

    def transcribe(
        self,
        audio: np.ndarray,
        task: str,
        language: Optional[str],
        word_timestamps: bool,
        output: str,
        options: Optional[dict] = None,
    ) -> Union[TranscriptionResponse, str]:
        """Транскрибирует аудио через Parakeet (вызов уходит в поток инференса)."""
        self.update_activity()
        self._start_worker()

        if task == "translate":
            logger.warning("Parakeet does not support translation; doing transcription")
        # word_timestamps: Parakeet отдаёт токены всегда, слова собираются бесплатно.

        audio = normalize_audio(audio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        duration = get_audio_duration(audio)

        done = threading.Event()
        holder: dict = {}
        self._jobs.put((audio, duration, done, holder))
        done.wait()

        if "error" in holder:
            raise holder["error"]
        return self._format_result(holder["result"], duration=duration, output=output)

    # --------------------------------------------------------------- inference

    def _run_transcribe(self, model, audio: np.ndarray, duration: float):
        """Прогоняет аудио через модель; короткое - целиком, длинное - окнами."""
        import soundfile as sf

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp, audio, 16000, subtype="PCM_16")
                tmp_path = tmp.name

            # Короткие чанки митингов идут целиком: оконный режим на них только
            # добавляет перекрытие и лишние проходы энкодера.
            chunk_duration = PARAKEET_CHUNK_SEC if duration > PARAKEET_CHUNK_SEC else None
            return model.transcribe(
                tmp_path,
                chunk_duration=chunk_duration,
                overlap_duration=PARAKEET_OVERLAP_SEC,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _format_result(self, result, duration: float, output: str) -> Union[TranscriptionResponse, str]:
        """Собирает TranscriptionResponse из AlignedResult."""
        texts: List[str] = []
        segments: List[Segment] = []

        for idx, sentence in enumerate(result.sentences):
            text = (sentence.text or "").strip()
            if not text:
                continue
            start = float(sentence.start or 0.0)
            end = float(sentence.end or 0.0)
            dur = end - start
            segments.append(
                Segment(
                    id=idx,
                    start=start,
                    end=end,
                    text=text,
                    words=_tokens_to_words(sentence.tokens) or None,
                    avg_word_score=(
                        round(float(sentence.confidence), 4)
                        if sentence.confidence is not None
                        else None
                    ),
                    chars_per_second=round(len(text) / dur, 4) if dur > 0 else None,
                )
            )
            texts.append(text)

        full_text = " ".join(texts).strip()
        if output == "text":
            return full_text

        response = TranscriptionResponse(
            text=full_text,
            # Parakeet мультиязычный с автоопределением языка - не навязываем ru.
            language=None,
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
                logger.debug("Failed to delete Parakeet model", exc_info=True)
            finally:
                self.model = None
