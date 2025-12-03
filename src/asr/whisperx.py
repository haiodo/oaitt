"""
OAITT — Open AI Transformer Transcriber.

ASR реализация на WhisperX.
Использует WhisperX для распознавания речи с точным выравниванием
временных меток слов на основе фонемной модели.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from threading import Thread
from typing import Optional, Union

import numpy as np

from src.asr.base import ASRModel
from src.config import (
    COMPUTE_TYPE,
    CONFIDENCE_AVG_LOGPROB_THRESHOLD,
    CONFIDENCE_LOW_PROB_RATIO_THRESHOLD,
    CONFIDENCE_NO_SPEECH_THRESHOLD,
    CONFIDENCE_WORD_PROB_THRESHOLD,
    CONFIDENCE_WORD_SCORE_THRESHOLD,
    MODEL_CACHE_DIR,
    MODEL_IDLE_TIMEOUT,
    WHISPERX_MODEL,
)
from src.models.schemas import (
    ConfidenceMetrics,
    Segment,
    TranscriptionResponse,
    WordTimestamp,
)
from src.utils.device import get_device, is_mps_device

logger = logging.getLogger(__name__)


class WhisperXASR(ASRModel):
    """
    ASR реализация на WhisperX.

    WhisperX обеспечивает более точные временные метки слов по сравнению
    с базовым Whisper благодаря использованию фонемной модели выравнивания.

    Особенности:
    - Точные временные метки на уровне слов
    - Автоматическое определение языка
    - Метрики уверенности для оценки качества
    - Поддержка множества языков для выравнивания

    Note:
        WhisperX использует ctranslate2, который не поддерживает MPS (Apple Silicon).
        На устройствах с MPS автоматически используется CPU.

    Attributes:
        model: Словарь с загруженными моделями:
            - 'whisperx': Основная модель транскрипции
            - 'align_model': Кэш моделей выравнивания по языкам

    Example:
        >>> asr = WhisperXASR()
        >>> asr.load_model()
        >>> result = asr.transcribe(
        ...     audio=audio_data,
        ...     task="transcribe",
        ...     language="en",
        ...     word_timestamps=True,
        ...     output="json",
        ... )
        >>> print(result.text)
        >>> print(result.confidence.avg_word_score)
    """

    def __init__(self):
        """Инициализирует WhisperXASR."""
        super().__init__()
        self.model = {
            "whisperx": None,
            "align_model": {},  # Cache for alignment models by language
        }

    def load_model(self) -> None:
        """
        Загружает модель WhisperX.

        Загружает основную модель транскрипции. Модели выравнивания
        загружаются лениво при первом использовании для каждого языка.

        Note:
            Если обнаружено MPS устройство, используется CPU,
            так как ctranslate2 не поддерживает MPS.

        Raises:
            Exception: При ошибке загрузки модели.
        """
        import whisperx

        device = get_device()

        # WhisperX (ctranslate2) doesn't support MPS directly, fall back to CPU
        if is_mps_device(device):
            device_str = "cpu"
            compute_type = "float32"  # CPU works best with float32
            logger.info(
                f"MPS detected, but WhisperX doesn't support MPS. "
                f"Using CPU with {compute_type}"
            )
        else:
            device_str = str(device)
            compute_type = COMPUTE_TYPE

        logger.info(
            f"Loading WhisperX model '{WHISPERX_MODEL}' "
            f"on device: {device_str}, compute_type: {compute_type}"
        )

        asr_options = {"without_timestamps": False}
        download_root = MODEL_CACHE_DIR if MODEL_CACHE_DIR else None

        self.model["whisperx"] = whisperx.load_model(
            WHISPERX_MODEL,
            device=device_str,
            compute_type=compute_type,
            asr_options=asr_options,
            download_root=download_root,
        )

        logger.info("WhisperX model loaded successfully")

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
        Выполняет транскрипцию аудио с выравниванием слов.

        Процесс:
        1. Транскрипция с помощью WhisperX
        2. Определение языка (если не указан)
        3. Выравнивание временных меток слов с помощью фонемной модели
        4. Вычисление метрик уверенности

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
        import whisperx

        self.update_activity()

        # Ensure model is loaded
        self.ensure_model_loaded()

        device = get_device()
        # WhisperX alignment also doesn't support MPS
        device_str = "cpu" if is_mps_device(device) else str(device)

        # Transcribe
        transcribe_options = {"task": task}
        if language:
            transcribe_options["language"] = language

        with self.model_lock:
            result = self.model["whisperx"].transcribe(audio, **transcribe_options)
            detected_language = result.get("language", language)

        # Save segment confidence metrics before alignment (alignment may lose them)
        segment_confidence = {}
        for idx, seg in enumerate(result.get("segments", [])):
            segment_confidence[idx] = {
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "temperature": seg.get("temperature"),
                "tokens": seg.get("tokens", []),
            }

        # Perform alignment for word-level timestamps
        try:
            if detected_language in self.model["align_model"]:
                model_x, metadata = self.model["align_model"][detected_language]
            else:
                logger.info(f"Loading alignment model for language: {detected_language}")
                self.model["align_model"][detected_language] = whisperx.load_align_model(
                    language_code=detected_language, device=device_str
                )
                model_x, metadata = self.model["align_model"][detected_language]

            result = whisperx.align(
                result["segments"],
                model_x,
                metadata,
                audio,
                device_str,
                return_char_alignments=False,
            )

            # Restore confidence metrics after alignment
            for idx, seg in enumerate(result.get("segments", [])):
                if idx in segment_confidence:
                    conf = segment_confidence[idx]
                    if conf["avg_logprob"] is not None:
                        seg["avg_logprob"] = conf["avg_logprob"]
                    if conf["no_speech_prob"] is not None:
                        seg["no_speech_prob"] = conf["no_speech_prob"]
                    if conf["compression_ratio"] is not None:
                        seg["compression_ratio"] = conf["compression_ratio"]
                    if conf["temperature"] is not None:
                        seg["temperature"] = conf["temperature"]
                    if conf["tokens"]:
                        seg["tokens"] = conf["tokens"]

        except Exception as e:
            logger.warning(f"Alignment failed for language '{detected_language}': {e}")

        result["language"] = detected_language

        logger.info(
            f"WhisperX transcription completed: "
            f"{len(result.get('segments', []))} segments"
        )

        return self._format_result(result, output)

    def _format_result(
        self,
        result: dict,
        output: str,
    ) -> Union[TranscriptionResponse, str]:
        """
        Форматирует результат WhisperX.

        Args:
            result: Сырой результат от WhisperX.
            output: Формат вывода.

        Returns:
            Отформатированный результат.
        """
        segments = result.get("segments", [])
        full_text = " ".join(seg.get("text", "").strip() for seg in segments)

        # Simple text output
        if output == "text":
            return full_text

        # Subtitle formats
        if output == "srt":
            return self._to_srt(segments)

        if output == "vtt":
            return self._to_vtt(segments)

        if output == "tsv":
            return self._to_tsv(segments)

        # JSON output - collect confidence metrics
        response_segments = []
        all_word_scores = []  # Alignment scores
        all_word_probs = []  # Model probabilities
        all_avg_logprobs = []
        all_no_speech_probs = []

        for idx, seg in enumerate(segments):
            words = None
            segment_word_scores = []

            if "words" in seg:
                words = []
                for w in seg["words"]:
                    score = w.get("score")  # Alignment score
                    prob = w.get("probability", w.get("prob"))  # Model probability

                    words.append(
                        WordTimestamp(
                            word=w.get("word", ""),
                            start=w.get("start", 0),
                            end=w.get("end", 0),
                            probability=score if score is not None else prob,
                        )
                    )

                    if score is not None:
                        segment_word_scores.append(score)
                        all_word_scores.append(score)
                    if prob is not None:
                        all_word_probs.append(prob)

            # Extract segment-level confidence from faster-whisper output
            avg_logprob = seg.get("avg_logprob")
            no_speech_prob = seg.get("no_speech_prob")

            if avg_logprob is not None:
                all_avg_logprobs.append(avg_logprob)
            if no_speech_prob is not None:
                all_no_speech_probs.append(no_speech_prob)

            # Calculate average word score for segment
            avg_word_score = (
                sum(segment_word_scores) / len(segment_word_scores)
                if segment_word_scores
                else None
            )

            response_segments.append(
                Segment(
                    id=idx,
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", "").strip(),
                    words=words,
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                    avg_word_score=avg_word_score,
                    compression_ratio=seg.get("compression_ratio"),
                    temperature=seg.get("temperature"),
                    tokens=seg.get("tokens"),
                )
            )

        # Calculate overall confidence metrics
        confidence = self._calculate_confidence_metrics(
            all_avg_logprobs, all_no_speech_probs, all_word_scores, all_word_probs
        )

        logger.info(
            f"Confidence metrics: avg_logprob={confidence.avg_logprob}, "
            f"no_speech_prob={confidence.no_speech_prob}, "
            f"avg_word_score={confidence.avg_word_score}, "
            f"avg_word_prob={confidence.avg_word_prob}, "
            f"min_word_prob={confidence.min_word_prob}, "
            f"low_conf_ratio={confidence.low_confidence_word_ratio}, "
            f"low_prob_ratio={confidence.low_prob_word_ratio}, "
            f"reliable={confidence.is_reliable}"
        )

        return TranscriptionResponse(
            text=full_text,
            language=result.get("language"),
            segments=response_segments,
            confidence=confidence,
        )

    def _calculate_confidence_metrics(
        self,
        avg_logprobs: list[float],
        no_speech_probs: list[float],
        word_scores: list[float],
        word_probs: Optional[list[float]] = None,
    ) -> ConfidenceMetrics:
        """
        Вычисляет общие метрики уверенности и определяет надёжность.

        Args:
            avg_logprobs: Средние log-вероятности из сегментов.
            no_speech_probs: Вероятности отсутствия речи из сегментов.
            word_scores: Оценки выравнивания слов (от модели выравнивания, 0-1).
            word_probs: Вероятности слов от модели транскрипции (0-1).

        Returns:
            ConfidenceMetrics с вычисленными метриками и флагом надёжности.
        """
        rejection_reasons = []
        word_probs = word_probs or []

        # Calculate averages
        avg_logprob = (
            sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None
        )
        no_speech_prob = (
            sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else None
        )
        avg_word_score = (
            sum(word_scores) / len(word_scores) if word_scores else None
        )

        # Calculate word probability statistics
        avg_word_prob = sum(word_probs) / len(word_probs) if word_probs else None
        min_word_prob = min(word_probs) if word_probs else None
        max_word_prob = max(word_probs) if word_probs else None

        # Calculate ratio of low-confidence words (alignment score)
        low_conf_words = sum(
            1 for s in word_scores if s < CONFIDENCE_WORD_SCORE_THRESHOLD
        )
        low_confidence_word_ratio = (
            low_conf_words / len(word_scores) if word_scores else None
        )

        # Calculate ratio of low-probability words (model confidence)
        low_prob_words = sum(
            1 for p in word_probs if p < CONFIDENCE_WORD_PROB_THRESHOLD
        )
        low_prob_word_ratio = (
            low_prob_words / len(word_probs) if word_probs else None
        )

        # Check reliability thresholds
        is_reliable = True

        if avg_logprob is not None and avg_logprob < CONFIDENCE_AVG_LOGPROB_THRESHOLD:
            is_reliable = False
            rejection_reasons.append(
                f"avg_logprob={avg_logprob:.3f} < threshold={CONFIDENCE_AVG_LOGPROB_THRESHOLD}"
            )

        if no_speech_prob is not None and no_speech_prob > CONFIDENCE_NO_SPEECH_THRESHOLD:
            is_reliable = False
            rejection_reasons.append(
                f"no_speech_prob={no_speech_prob:.3f} > threshold={CONFIDENCE_NO_SPEECH_THRESHOLD}"
            )

        if avg_word_score is not None and avg_word_score < CONFIDENCE_WORD_SCORE_THRESHOLD:
            is_reliable = False
            rejection_reasons.append(
                f"avg_word_score={avg_word_score:.3f} < threshold={CONFIDENCE_WORD_SCORE_THRESHOLD}"
            )

        if avg_word_prob is not None and avg_word_prob < CONFIDENCE_WORD_PROB_THRESHOLD:
            is_reliable = False
            rejection_reasons.append(
                f"avg_word_prob={avg_word_prob:.3f} < threshold={CONFIDENCE_WORD_PROB_THRESHOLD}"
            )

        if (
            low_prob_word_ratio is not None
            and low_prob_word_ratio > CONFIDENCE_LOW_PROB_RATIO_THRESHOLD
        ):
            is_reliable = False
            rejection_reasons.append(
                f"low_prob_word_ratio={low_prob_word_ratio:.3f} > threshold={CONFIDENCE_LOW_PROB_RATIO_THRESHOLD}"
            )

        return ConfidenceMetrics(
            avg_logprob=round(avg_logprob, 4) if avg_logprob is not None else None,
            no_speech_prob=round(no_speech_prob, 4) if no_speech_prob is not None else None,
            avg_word_score=round(avg_word_score, 4) if avg_word_score is not None else None,
            avg_word_prob=round(avg_word_prob, 4) if avg_word_prob is not None else None,
            min_word_prob=round(min_word_prob, 4) if min_word_prob is not None else None,
            max_word_prob=round(max_word_prob, 4) if max_word_prob is not None else None,
            low_confidence_word_ratio=(
                round(low_confidence_word_ratio, 4)
                if low_confidence_word_ratio is not None
                else None
            ),
            low_prob_word_ratio=(
                round(low_prob_word_ratio, 4) if low_prob_word_ratio is not None else None
            ),
            is_reliable=is_reliable,
            rejection_reasons=rejection_reasons if rejection_reasons else None,
        )

    def _to_srt(self, segments: list) -> str:
        """
        Форматирует сегменты в SRT формат.

        Args:
            segments: Список сегментов от WhisperX.

        Returns:
            Строка в формате SRT.
        """
        lines = []
        for idx, seg in enumerate(segments):
            start = self._format_timestamp_srt(seg.get("start", 0))
            end = self._format_timestamp_srt(seg.get("end", 0))
            lines.append(str(idx + 1))
            lines.append(f"{start} --> {end}")
            lines.append(seg.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    def _to_vtt(self, segments: list) -> str:
        """
        Форматирует сегменты в WebVTT формат.

        Args:
            segments: Список сегментов от WhisperX.

        Returns:
            Строка в формате WebVTT.
        """
        lines = ["WEBVTT", ""]
        for seg in segments:
            start = self._format_timestamp_vtt(seg.get("start", 0))
            end = self._format_timestamp_vtt(seg.get("end", 0))
            lines.append(f"{start} --> {end}")
            lines.append(seg.get("text", "").strip())
            lines.append("")
        return "\n".join(lines)

    def _to_tsv(self, segments: list) -> str:
        """
        Форматирует сегменты в TSV формат.

        Args:
            segments: Список сегментов от WhisperX.

        Returns:
            Строка в формате TSV.
        """
        lines = ["start\tend\ttext"]
        for seg in segments:
            start = int(seg.get("start", 0) * 1000)
            end = int(seg.get("end", 0) * 1000)
            text = seg.get("text", "").strip().replace("\t", " ")
            lines.append(f"{start}\t{end}\t{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Форматирует секунды в SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Форматирует секунды в VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _cleanup_model(self) -> None:
        """Очищает ресурсы WhisperX."""
        if self.model is not None:
            if self.model.get("whisperx") is not None:
                del self.model["whisperx"]
                self.model["whisperx"] = None

            if self.model.get("align_model"):
                self.model["align_model"].clear()
