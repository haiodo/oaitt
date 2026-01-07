"""
OAITT — Open AI Transformer Transcriber.

OpenAI-совместимые модели для API.
Эти модели соответствуют формату ответов OpenAI Audio Transcriptions API,
что позволяет использовать сервис как drop-in замену OpenAI Whisper API.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from typing import Optional
from pydantic import BaseModel


class OpenAIWord(BaseModel):
    """
    OpenAI формат временной метки слова.

    Соответствует формату word из OpenAI verbose_json ответа.
    """
    word: str
    start: float
    end: float
    prob: Optional[float] = None  # Word probability/confidence (0-1)


class OpenAISegment(BaseModel):
    """
    OpenAI формат сегмента транскрипции.

    Соответствует формату segment из OpenAI verbose_json ответа.
    """
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = -1.0  # Default to -1 if not available
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0

    # Characters per second observed for the segment (len(text) / (end - start))
    chars_per_second: Optional[float] = None


class OpenAITranscriptionResponse(BaseModel):
    """
    OpenAI-совместимый ответ транскрипции (verbose_json формат).

    Соответствует формату ответа OpenAI Audio Transcriptions API
    с response_format=verbose_json.

    Пример ответа:
    {
        "text": "Hello world",
        "task": "transcribe",
        "language": "en",
        "duration": 2.5,
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "prob": 0.95},
            {"word": "world", "start": 0.6, "end": 1.0, "prob": 0.92}
        ],
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.3,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.01
            }
        ]
    }
    """
    text: str
    task: str = "transcribe"
    language: Optional[str] = None
    duration: Optional[float] = None

    # Overall characters per second observed for the transcription (len(text) / duration)
    chars_per_second: Optional[float] = None

    words: Optional[list[OpenAIWord]] = None
    segments: Optional[list[OpenAISegment]] = None
