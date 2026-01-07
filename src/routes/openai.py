"""
OAITT — Open AI Transformer Transcriber.

OpenAI-совместимый маршрут транскрипции.
Предоставляет эндпоинт /v1/audio/transcriptions, который совместим
с OpenAI Audio Transcriptions API, позволяя использовать сервис
как drop-in замену OpenAI Whisper API.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.auth import verify_token

from src.config import (
    SAMPLE_RATE,
    ASR_ENGINE,
    CONFIDENCE_FILTER_ENABLED,
    MAX_CHARS_PER_SECOND,
    CHARS_PER_SECOND_MULTIPLIER,
    CHARS_PER_SECOND_MIN_AUDIO_SEC,
)
from src.models.openai import (
    OpenAISegment,
    OpenAITranscriptionResponse,
    OpenAIWord,
)
from src.models.schemas import TranscriptionResponse, ConfidenceMetrics
from src.services.debug import save_debug_log
from src.services.timeout import TranscriptionTimeoutError, transcribe_with_timeout
from src.utils.audio import load_audio_from_file
from src.utils.formatters import format_srt, format_vtt

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI Compatible"])

# Reference to global ASR model (will be set by app)
_asr_model: "ASRModel | None" = None


def set_asr_model(model: "ASRModel") -> None:
    """
    Устанавливает ссылку на глобальную ASR модель.

    Args:
        model: Экземпляр ASR модели.
    """
    global _asr_model
    _asr_model = model


@router.post("/v1/audio/transcriptions")
async def openai_transcribe(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    _token: str = Depends(verify_token),
    model: str = Form("whisper-1", description="Model name (ignored, uses configured model)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'ru')"),
    prompt: Optional[str] = Form(None, description="Optional prompt (not used)"),
    response_format: str = Form("json", description="Response format: json, text, srt, vtt, verbose_json"),
    temperature: float = Form(0.0, description="Temperature (not used)"),
    timestamp_granularities: Optional[list[str]] = Form(None, description="Timestamp granularities: word, segment"),
):
    """
    OpenAI-совместимый эндпоинт транскрипции.

    Этот эндпоинт имитирует OpenAI Audio Transcriptions API:
    POST https://api.openai.com/v1/audio/transcriptions

    Поддерживаемые форматы ответа:
    - json: Простой JSON с полем text
    - text: Обычный текст
    - srt: SubRip формат субтитров
    - vtt: WebVTT формат субтитров
    - verbose_json: Полный JSON со словами, сегментами и метаданными

    Note:
        Параметр 'model' принимается, но игнорируется - используется
        настроенный ASR движок.

    Args:
        file: Аудиофайл для транскрипции.
        model: Название модели (игнорируется).
        language: Код языка (например, "en", "ru").
        prompt: Необязательная подсказка (не используется).
        response_format: Формат ответа.
        temperature: Температура (не используется).
        timestamp_granularities: Гранулярность временных меток: "word", "segment".

    Returns:
        Результат транскрипции в выбранном формате.

    Raises:
        HTTPException(503): Если модель не загружена.
        HTTPException(408): Если превышен таймаут транскрипции.
        HTTPException(400): Если указан неподдерживаемый формат ответа.
        HTTPException(500): При внутренней ошибке.

    Example:
        ```bash
        curl -X POST "http://localhost:9007/v1/audio/transcriptions" \\
          -F "file=@audio.wav" \\
          -F "model=whisper-1" \\
          -F "response_format=verbose_json" \\
          -F "timestamp_granularities[]=word" \\
          -F "timestamp_granularities[]=segment"
        ```
    """
    if _asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Determine if word timestamps are needed
    want_words = response_format == "verbose_json" or (
        timestamp_granularities is not None and "word" in timestamp_granularities
    )

    try:
        # Read audio file
        audio_content = await file.read()
        logger.info(
            f"[OpenAI API] Received audio file: {file.filename}, "
            f"size: {len(audio_content)} bytes"
        )

        # Convert audio to numpy array
        audio_data = load_audio_from_file(audio_content)
        audio_duration_sec = len(audio_data) / SAMPLE_RATE
        logger.info(
            f"[OpenAI API] Audio converted: {len(audio_data)} samples, "
            f"duration: {audio_duration_sec:.2f}s"
        )

        # Use internal output format based on what we need
        internal_output = "json" if want_words or response_format == "verbose_json" else "text"

        # Run transcription with adaptive timeout
        try:
            result, elapsed_time = transcribe_with_timeout(
                asr_model=_asr_model,
                audio=audio_data,
                audio_duration_sec=audio_duration_sec,
                task="transcribe",
                language=language,
                word_timestamps=want_words,
                output=internal_output,
            )
        except TranscriptionTimeoutError as e:
            logger.error(f"[OpenAI API] Transcription timeout: {e}")
            raise HTTPException(
                status_code=408,
                detail=(
                    f"Transcription timed out after {e.elapsed:.1f}s "
                    f"(expected {e.expected:.1f}s). "
                    f"This may indicate audio issues or model hallucination."
                ),
            )

        speed_ratio = audio_duration_sec / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"[OpenAI API] Transcription completed: duration={elapsed_time:.3f}s, "
            f"audio={audio_duration_sec:.2f}s, speed={speed_ratio:.1f}x realtime"
        )

        # Save debug log if enabled
        save_debug_log(audio_data, result, file.filename)

        # Compute characters-per-second for the whole transcription and optionally filter
        if isinstance(result, TranscriptionResponse):
            total_chars = len(result.text) if result.text else 0
            chars_per_sec = total_chars / audio_duration_sec if audio_duration_sec and audio_duration_sec > 0 else None

            # Ensure ConfidenceMetrics exists for augmentation
            if result.confidence is None:
                result.confidence = ConfidenceMetrics()

            # Store per-response and per-confidence values
            result.chars_per_second = round(chars_per_sec, 4) if chars_per_sec is not None else None
            result.confidence.chars_per_second = round(chars_per_sec, 4) if chars_per_sec is not None else None

            # Compute threshold and ratio for diagnostics
            threshold = MAX_CHARS_PER_SECOND * CHARS_PER_SECOND_MULTIPLIER
            result.confidence.chars_per_second_threshold = round(threshold, 4)
            if MAX_CHARS_PER_SECOND and chars_per_sec is not None:
                result.confidence.chars_per_second_ratio = round(chars_per_sec / MAX_CHARS_PER_SECOND, 4)
            else:
                result.confidence.chars_per_second_ratio = None

            # If observed chars/sec is many times above baseline, mark as suspicious
            if (
                chars_per_sec is not None
                and audio_duration_sec >= CHARS_PER_SECOND_MIN_AUDIO_SEC
                and MAX_CHARS_PER_SECOND > 0
                and result.confidence.chars_per_second_ratio is not None
                and result.confidence.chars_per_second_ratio > CHARS_PER_SECOND_MULTIPLIER
            ):
                result.confidence.high_char_rate = True
                result.confidence.is_reliable = False
                if not result.confidence.rejection_reasons:
                    result.confidence.rejection_reasons = []
                result.confidence.rejection_reasons.append(
                    f"chars_per_second={chars_per_sec:.2f} > threshold={threshold:.2f}"
                )
                logger.warning(f"[OpenAI API] High characters/sec detected: {chars_per_sec:.2f} chars/s (threshold={threshold:.2f})")

                if CONFIDENCE_FILTER_ENABLED:
                    logger.info("[OpenAI API] Filtering out high char-rate result (returning empty)")
                    # Return empty responses consistent with requested response_format
                    if response_format == "text":
                        return PlainTextResponse(content="")
                    elif response_format == "json":
                        return JSONResponse(
                            content={"text": ""},
                            headers={"Asr-Engine": ASR_ENGINE, "X-Confidence-Filtered": "true"},
                        )
                    elif response_format == "verbose_json":
                        # Return an empty verbose JSON response compatible with OpenAI
                        empty = OpenAITranscriptionResponse(
                            text="", task="transcribe", language=language, duration=audio_duration_sec
                        )
                        return JSONResponse(
                            content=empty.model_dump(exclude_none=True),
                            headers={"Asr-Engine": ASR_ENGINE, "X-Confidence-Filtered": "true"},
                        )
                    elif response_format == "srt":
                        return PlainTextResponse(content="", media_type="text/plain")
                    elif response_format == "vtt":
                        return PlainTextResponse(content="WEBVTT\n\n", media_type="text/vtt")
                    else:
                        return PlainTextResponse(content="")

        # Format response based on response_format
        return _format_openai_response(
            result=result,
            response_format=response_format,
            language=language,
            audio_duration_sec=audio_duration_sec,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[OpenAI API] Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _format_openai_response(
    result: TranscriptionResponse | str,
    response_format: str,
    language: Optional[str],
    audio_duration_sec: float,
):
    """
    Форматирует ответ в OpenAI-совместимом формате.

    Args:
        result: Результат транскрипции.
        response_format: Формат ответа.
        language: Код языка.
        audio_duration_sec: Длительность аудио в секундах.

    Returns:
        Отформатированный HTTP ответ.

    Raises:
        HTTPException: Если формат ответа не поддерживается.
    """
    if response_format == "text":
        text = result if isinstance(result, str) else result.text
        return PlainTextResponse(content=text)

    elif response_format == "srt":
        if isinstance(result, str):
            return PlainTextResponse(content="", media_type="text/plain")
        return PlainTextResponse(
            content=format_srt(result),
            media_type="text/plain",
        )

    elif response_format == "vtt":
        if isinstance(result, str):
            return PlainTextResponse(content="WEBVTT\n\n", media_type="text/vtt")
        return PlainTextResponse(
            content=format_vtt(result),
            media_type="text/vtt",
        )

    elif response_format == "json":
        # Simple JSON format (OpenAI default)
        text = result if isinstance(result, str) else result.text
        return JSONResponse(content={"text": text})

    elif response_format == "verbose_json":
        # Full verbose JSON with words and segments
        if isinstance(result, str):
            return JSONResponse(
                content={
                    "text": result,
                    "task": "transcribe",
                    "language": language,
                    "duration": audio_duration_sec,
                }
            )

        # Build OpenAI-compatible response and include overall chars/sec if present
        openai_response = OpenAITranscriptionResponse(
            text=result.text,
            task="transcribe",
            language=result.language or language,
            duration=audio_duration_sec,
            chars_per_second=result.chars_per_second,
        )

        # Add words and segments if available
        if result.segments:
            openai_words = []
            openai_segments = []

            for seg in result.segments:
                # Add segment with confidence metrics and chars/sec
                openai_seg = OpenAISegment(
                    id=seg.id,
                    seek=0,  # Not tracked currently
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    tokens=seg.tokens if seg.tokens else [],
                    temperature=seg.temperature if seg.temperature is not None else 0.0,
                    avg_logprob=seg.avg_logprob if seg.avg_logprob is not None else -1.0,
                    compression_ratio=(
                        seg.compression_ratio if seg.compression_ratio is not None else 0.0
                    ),
                    no_speech_prob=(
                        seg.no_speech_prob if seg.no_speech_prob is not None else 0.0
                    ),
                    chars_per_second=seg.chars_per_second,
                )
                openai_segments.append(openai_seg)

                # Add words from segment
                if seg.words:
                    for w in seg.words:
                        openai_words.append(
                            OpenAIWord(
                                word=w.word,
                                start=w.start,
                                end=w.end,
                                prob=w.probability,  # Include word confidence
                            )
                        )

            if openai_words:
                openai_response.words = openai_words
            if openai_segments:
                openai_response.segments = openai_segments

        return JSONResponse(content=openai_response.model_dump(exclude_none=True))

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {response_format}",
        )
