"""
OAITT — Open AI Transformer Transcriber.

Pydantic модели для транскрипции.
Содержит схемы данных для результатов распознавания речи,
включая временные метки слов, сегменты и метрики уверенности.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from typing import Optional
from pydantic import BaseModel


class WordTimestamp(BaseModel):
    """Временная метка слова с опциональной вероятностью."""
    word: str
    start: float
    end: float
    probability: Optional[float] = None


class Segment(BaseModel):
    """
    Сегмент транскрипции.

    Представляет собой фрагмент распознанного текста с временными
    метками и опциональными метриками качества.
    """
    id: int
    start: float
    end: float
    text: str
    words: Optional[list[WordTimestamp]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    avg_word_score: Optional[float] = None  # Average word alignment confidence (0-1)
    compression_ratio: Optional[float] = None
    temperature: Optional[float] = None
    tokens: Optional[list[int]] = None


class ConfidenceMetrics(BaseModel):
    """
    Метрики уверенности для всей транскрипции.

    Используется для оценки качества распознавания и фильтрации
    ненадёжных результатов.
    """
    # Average log probability across all segments
    avg_logprob: Optional[float] = None

    # Average no-speech probability
    no_speech_prob: Optional[float] = None

    # Average word alignment score (from alignment model)
    avg_word_score: Optional[float] = None

    # Average word probability from transcription model (0-1)
    avg_word_prob: Optional[float] = None

    # Minimum word probability (worst word)
    min_word_prob: Optional[float] = None

    # Maximum word probability (best word)
    max_word_prob: Optional[float] = None

    # Ratio of words with score < threshold
    low_confidence_word_ratio: Optional[float] = None

    # Ratio of words with prob < threshold
    low_prob_word_ratio: Optional[float] = None

    # Whether transcription passes confidence thresholds
    is_reliable: bool = True

    # Reasons if is_reliable=False
    rejection_reasons: Optional[list[str]] = None


class TranscriptionResponse(BaseModel):
    """
    Ответ с результатом транскрипции.

    Основная модель ответа для всех эндпоинтов транскрипции.
    """
    text: str
    language: Optional[str] = None
    segments: Optional[list[Segment]] = None
    confidence: Optional[ConfidenceMetrics] = None
