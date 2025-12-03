"""
OAITT — Open AI Transformer Transcriber.

Маршрут основного ASR эндпоинта.
Предоставляет эндпоинт /asr для транскрипции аудиофайлов
с поддержкой различных форматов вывода.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import ASR_ENGINE, CONFIDENCE_FILTER_ENABLED, SAMPLE_RATE
from src.models.schemas import TranscriptionResponse
from src.services.debug import save_debug_log
from src.services.timeout import TranscriptionTimeoutError, transcribe_with_timeout
from src.utils.audio import load_audio_from_file
from src.utils.formatters import format_srt, format_tsv, format_vtt

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ASR"])

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


@router.post("/asr")
async def transcribe(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    output: str = Query("json", description="Output format: text, json, vtt, srt, tsv"),
    task: str = Query("transcribe", description="Task: transcribe or translate"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'russian', 'en')"),
    word_timestamps: bool = Query(True, description="Include word-level timestamps"),
    encode: bool = Query(False, description="Whether audio needs encoding (ignored, handled automatically)"),
):
    """
    Транскрибирует аудиофайл с помощью Whisper модели.

    Поддерживает два движка:
    - transformers: Hugging Face Transformers pipeline (по умолчанию)
    - whisperx: WhisperX с выравниванием слов

    Для выбора движка установите переменную окружения ASR_ENGINE.

    Args:
        audio_file: Аудиофайл для транскрипции.
        output: Формат вывода:
            - "text": Только текст
            - "json": JSON с сегментами и метаданными
            - "vtt": WebVTT субтитры
            - "srt": SRT субтитры
            - "tsv": Tab-separated values
        task: Задача - "transcribe" или "translate".
        language: Код языка (например, "en", "ru", "russian").
        word_timestamps: Включить временные метки слов.
        encode: Устаревший параметр (игнорируется).

    Returns:
        Результат транскрипции в выбранном формате.

    Raises:
        HTTPException(503): Если модель не загружена.
        HTTPException(408): Если превышен таймаут транскрипции.
        HTTPException(400): Если указан неподдерживаемый формат вывода.
        HTTPException(500): При внутренней ошибке.
    """
    if _asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read audio file
        audio_content = await audio_file.read()
        logger.info(
            f"Received audio file: {audio_file.filename}, "
            f"size: {len(audio_content)} bytes"
        )

        # Convert audio to numpy array
        audio_data = load_audio_from_file(audio_content)
        audio_duration_sec = len(audio_data) / SAMPLE_RATE
        logger.info(
            f"Audio converted: {len(audio_data)} samples at {SAMPLE_RATE}Hz, "
            f"duration: {audio_duration_sec:.2f}s"
        )

        # Run transcription with adaptive timeout
        try:
            result, elapsed_time = transcribe_with_timeout(
                asr_model=_asr_model,
                audio=audio_data,
                audio_duration_sec=audio_duration_sec,
                task=task,
                language=language,
                word_timestamps=word_timestamps,
                output=output,
            )
        except TranscriptionTimeoutError as e:
            logger.error(f"Transcription timeout: {e}")
            raise HTTPException(
                status_code=408,
                detail=(
                    f"Transcription timed out after {e.elapsed:.1f}s "
                    f"(expected {e.expected:.1f}s). "
                    f"This may indicate audio issues or model hallucination."
                ),
            )

        # Calculate speed ratio (how many times faster than realtime)
        speed_ratio = audio_duration_sec / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"Transcription completed: duration={elapsed_time:.3f}s, "
            f"audio={audio_duration_sec:.2f}s, speed={speed_ratio:.1f}x realtime"
        )

        # Save debug log if enabled
        save_debug_log(audio_data, result, audio_file.filename)

        # Check confidence and optionally filter low-quality results
        if isinstance(result, TranscriptionResponse) and result.confidence:
            conf = result.confidence
            if not conf.is_reliable:
                reasons = (
                    ", ".join(conf.rejection_reasons)
                    if conf.rejection_reasons
                    else "unknown"
                )
                logger.warning(f"Low confidence transcription: {reasons}")

                if CONFIDENCE_FILTER_ENABLED:
                    logger.info("Filtering out low-confidence result (returning empty)")
                    # Return empty result for low-confidence transcriptions
                    if output == "text":
                        return PlainTextResponse(content="")
                    elif output == "json":
                        empty_result = TranscriptionResponse(
                            text="",
                            language=result.language,
                            segments=[],
                            confidence=conf,
                        )
                        return JSONResponse(
                            content=empty_result.model_dump(exclude_none=True),
                            headers={
                                "Asr-Engine": ASR_ENGINE,
                                "X-Confidence-Filtered": "true",
                            },
                        )
                    else:
                        return PlainTextResponse(content="")

        # Format response based on output type
        return _format_response(result, output)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _format_response(
    result: TranscriptionResponse | str,
    output: str,
):
    """
    Форматирует ответ в зависимости от типа вывода.

    Args:
        result: Результат транскрипции.
        output: Формат вывода.

    Returns:
        Отформатированный HTTP ответ.

    Raises:
        HTTPException: Если формат вывода не поддерживается.
    """
    if output == "text":
        if isinstance(result, str):
            return PlainTextResponse(content=result)
        return PlainTextResponse(content=result.text)

    elif output == "json":
        if isinstance(result, TranscriptionResponse):
            return JSONResponse(
                content=result.model_dump(exclude_none=True),
                headers={"Asr-Engine": ASR_ENGINE},
            )
        return JSONResponse(
            content={"text": str(result)},
            headers={"Asr-Engine": ASR_ENGINE},
        )

    elif output == "vtt":
        if isinstance(result, str):
            return PlainTextResponse(content=result, media_type="text/vtt")
        return PlainTextResponse(
            content=format_vtt(result),
            media_type="text/vtt",
        )

    elif output == "srt":
        if isinstance(result, str):
            return PlainTextResponse(content=result, media_type="text/plain")
        return PlainTextResponse(
            content=format_srt(result),
            media_type="text/plain",
        )

    elif output == "tsv":
        if isinstance(result, str):
            return PlainTextResponse(
                content=result,
                media_type="text/tab-separated-values",
            )
        return PlainTextResponse(
            content=format_tsv(result),
            media_type="text/tab-separated-values",
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported output format: {output}",
        )
