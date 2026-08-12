"""
OAITT — Open AI Transformer Transcriber.

Утилиты для загрузки и обработки аудио.
Предоставляет функции для загрузки аудиофайлов различных форматов
и их преобразования в формат, требуемый Whisper (16kHz mono float32).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import io
import logging
from typing import Tuple

import av
import numpy as np
import soundfile as sf
import librosa

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def _decode_with_av(audio_content: bytes) -> Tuple[np.ndarray, int]:
    """Декодирует любой контейнер, который понимает libav (webm/opus, ogg/opus, mp4/aac).

    In-process через PyAV: нет fork'а ffmpeg (~34ms на запуск), нет требования
    seekable stdin (mp4 с moov atom в конце через `ffmpeg -i pipe:0` не читается).
    Ресемплинг в mono float32 @ SAMPLE_RATE делает сам libav.
    """
    chunks = []
    with av.open(io.BytesIO(audio_content)) as container:
        if not container.streams.audio:
            raise RuntimeError("No audio stream in container")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise RuntimeError("Decoder produced no audio frames")
    return np.concatenate(chunks), SAMPLE_RATE


def load_audio_from_file(audio_content: bytes) -> np.ndarray:
    """
    Загружает аудио из байтов и преобразует в numpy array.

    Поддерживает различные форматы аудио (wav, flac, ogg, mp3 и др.)
    и автоматически преобразует в формат, требуемый Whisper:
    - Частота дискретизации: 16kHz
    - Каналы: моно
    - Тип данных: float32

    Args:
        audio_content: Байты аудиофайла.

    Returns:
        np.ndarray: Аудиоданные в формате float32 с частотой 16kHz.

    Raises:
        Exception: Если аудио не удалось загрузить ни одним из методов.
    """
    audio_buffer = io.BytesIO(audio_content)

    try:
        # soundfile первым: для wav/flac/ogg он в разы быстрее libav (0.7ms vs 6.6ms на 30s wav)
        audio_data, sample_rate = sf.read(audio_buffer)
        logger.debug(f"Audio loaded with soundfile: {sample_rate}Hz")
    except Exception as e:
        # libav покрывает контейнеры, которых libsndfile не знает: webm/opus (Chrome
        # MediaRecorder), mp4/aac (Safari), mp3 и т.д.
        logger.debug(f"soundfile failed ({e}), falling back to libav")
        audio_data, sample_rate = _decode_with_av(audio_content)

    # Free the BytesIO buffer immediately — no longer needed
    audio_buffer.close()
    del audio_buffer

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        stereo = audio_data
        audio_data = np.mean(stereo, axis=1)
        del stereo
        logger.debug("Converted stereo to mono")

    # Resample to 16kHz if needed (Whisper requirement)
    if sample_rate != SAMPLE_RATE:
        logger.debug(f"Resampling from {sample_rate}Hz to {SAMPLE_RATE}Hz")
        original = audio_data
        audio_data = librosa.resample(
            original,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )
        del original

    # Ensure float32 (avoid copy if already float32)
    if audio_data.dtype != np.float32:
        return audio_data.astype(np.float32)
    return audio_data


def get_audio_duration(audio_data: np.ndarray) -> float:
    """
    Вычисляет длительность аудио в секундах.

    Args:
        audio_data: Аудиоданные в формате numpy array (16kHz).

    Returns:
        Длительность аудио в секундах.
    """
    return len(audio_data) / SAMPLE_RATE


def load_audio_from_path(file_path: str) -> np.ndarray:
    """
    Загружает аудио из файла по пути.

    Args:
        file_path: Путь к аудиофайлу.

    Returns:
        np.ndarray: Аудиоданные в формате float32 с частотой 16kHz.
    """
    with open(file_path, "rb") as f:
        audio_content = f.read()
    return load_audio_from_file(audio_content)


def validate_audio(audio_data: np.ndarray) -> Tuple[bool, str]:
    """
    Валидирует аудиоданные.

    Проверяет:
    - Наличие данных
    - Допустимый диапазон значений
    - Минимальную длительность

    Args:
        audio_data: Аудиоданные для проверки.

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if audio_data is None or len(audio_data) == 0:
        return False, "Audio data is empty"

    duration = get_audio_duration(audio_data)
    if duration < 0.1:
        return False, f"Audio too short: {duration:.2f}s (minimum 0.1s)"

    if duration > 3600:
        return False, f"Audio too long: {duration:.2f}s (maximum 3600s)"

    # Check for valid audio range
    if np.max(np.abs(audio_data)) > 1.0:
        logger.warning("Audio values exceed [-1, 1] range, normalizing")

    return True, ""


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Нормализует аудиоданные.

    Приводит значения к диапазону [-1, 1] если они выходят за пределы.

    Args:
        audio_data: Аудиоданные для нормализации.

    Returns:
        np.ndarray: Нормализованные аудиоданные.
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        return audio_data / max_val
    return audio_data
