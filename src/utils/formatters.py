"""
OAITT — Open AI Transformer Transcriber.

Утилиты форматирования вывода.
Предоставляет функции для преобразования результатов транскрипции
в различные форматы субтитров и табличные форматы.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.schemas import TranscriptionResponse


def format_timestamp_vtt(seconds: float) -> str:
    """
    Форматирует секунды в VTT timestamp (HH:MM:SS.mmm).

    Args:
        seconds: Время в секундах.

    Returns:
        Строка в формате VTT timestamp.

    Example:
        >>> format_timestamp_vtt(3661.5)
        '01:01:01.500'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_srt(seconds: float) -> str:
    """
    Форматирует секунды в SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Время в секундах.

    Returns:
        Строка в формате SRT timestamp.

    Example:
        >>> format_timestamp_srt(3661.5)
        '01:01:01,500'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt(response: "TranscriptionResponse") -> str:
    """
    Форматирует TranscriptionResponse в WebVTT формат.

    WebVTT (Web Video Text Tracks) - стандартный формат субтитров
    для веб-браузеров.

    Args:
        response: Результат транскрипции.

    Returns:
        Строка в формате WebVTT.

    Example output:
        WEBVTT

        00:00:00.000 --> 00:00:02.500
        Hello world

        00:00:02.500 --> 00:00:05.000
        How are you?
    """
    lines = ["WEBVTT", ""]

    if response.segments:
        for seg in response.segments:
            start = format_timestamp_vtt(seg.start)
            end = format_timestamp_vtt(seg.end)
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")

    return "\n".join(lines)


def format_srt(response: "TranscriptionResponse") -> str:
    """
    Форматирует TranscriptionResponse в SubRip (SRT) формат.

    SRT - один из самых распространённых форматов субтитров,
    поддерживаемый большинством видеоплееров.

    Args:
        response: Результат транскрипции.

    Returns:
        Строка в формате SRT.

    Example output:
        1
        00:00:00,000 --> 00:00:02,500
        Hello world

        2
        00:00:02,500 --> 00:00:05,000
        How are you?
    """
    lines = []

    if response.segments:
        for idx, seg in enumerate(response.segments):
            start = format_timestamp_srt(seg.start)
            end = format_timestamp_srt(seg.end)
            lines.append(str(idx + 1))
            lines.append(f"{start} --> {end}")
            lines.append(seg.text)
            lines.append("")

    return "\n".join(lines)


def format_tsv(response: "TranscriptionResponse") -> str:
    """
    Форматирует TranscriptionResponse в TSV (Tab-Separated Values) формат.

    Табличный формат с временными метками в миллисекундах,
    удобный для импорта в электронные таблицы.

    Args:
        response: Результат транскрипции.

    Returns:
        Строка в формате TSV.

    Example output:
        start	end	text
        0	2500	Hello world
        2500	5000	How are you?
    """
    lines = ["start\tend\ttext"]

    if response.segments:
        for seg in response.segments:
            start = int(seg.start * 1000)
            end = int(seg.end * 1000)
            # Replace tabs in text to avoid breaking TSV format
            text = seg.text.replace("\t", " ")
            lines.append(f"{start}\t{end}\t{text}")

    return "\n".join(lines)
