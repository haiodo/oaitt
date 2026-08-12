#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Тесты декодирования всех форматов, которые реально приходят от клиентов:
  1. WebM / Opus, 48 kHz mono  — Chrome MediaRecorder
  2. Ogg  / Opus, 16 kHz mono  — love-agent; Firefox шлёт 48 kHz
  3. MP4  / AAC-LC, 44.1-48 kHz mono — Safari
  4. WAV  / PCM s16le          — audioFormat: 'wav' и /love/send_raw

Файлы генерируются на лету через ffmpeg из синтезированного тона.

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python -m pytest tests/test_audio_formats.py -v
"""

import shutil
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE
from src.utils.audio import load_audio_from_file

DURATION_SEC = 3.0
TONE_HZ = 440.0

# (имя, ffmpeg output args) — контейнер выводится из расширения
FORMATS = [
    ("chrome.webm", ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-f", "webm"]),
    ("love-agent.ogg", ["-ac", "1", "-ar", "16000", "-c:a", "libopus", "-f", "ogg"]),
    ("firefox.ogg", ["-ac", "1", "-ar", "48000", "-c:a", "libopus", "-f", "ogg"]),
    ("safari.mp4", ["-ac", "1", "-ar", "44100", "-c:a", "aac"]),
    ("raw.wav", ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]),
]


def _source_wav(tmp_path):
    """Тон 440 Гц, 48 kHz mono — источник для перекодирования."""
    t = np.linspace(0, DURATION_SEC, int(48000 * DURATION_SEC), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    path = tmp_path / "source.wav"
    sf.write(path, tone, 48000)
    return path


@pytest.fixture(scope="module")
def encoded(tmp_path_factory):
    """Кодирует источник во все клиентские форматы. Возвращает {name: bytes}."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg needed to generate fixtures (not to decode them)")

    tmp_path = tmp_path_factory.mktemp("formats")
    src = _source_wav(tmp_path)
    out = {}
    for name, args in FORMATS:
        dst = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), *args, str(dst)],
            check=True,
        )
        out[name] = dst.read_bytes()
    return out


@pytest.mark.parametrize("name", [n for n, _ in FORMATS])
def test_decode_format(encoded, name):
    """Каждый клиентский формат декодируется в 16 kHz mono float32 нужной длины."""
    audio = load_audio_from_file(encoded[name])

    assert audio.dtype == np.float32, f"{name}: dtype {audio.dtype}"
    assert audio.ndim == 1, f"{name}: не mono, shape {audio.shape}"

    # Opus/AAC добавляют priming samples — допуск 0.15s
    expected = DURATION_SEC * SAMPLE_RATE
    assert abs(len(audio) - expected) < 0.15 * SAMPLE_RATE, (
        f"{name}: {len(audio)} сэмплов, ожидалось ~{expected:.0f}"
    )

    # Не тишина и не мусор
    assert 0.05 < float(np.abs(audio).max()) <= 1.01, f"{name}: peak вне диапазона"


@pytest.mark.parametrize("name", [n for n, _ in FORMATS])
def test_decode_preserves_tone(encoded, name):
    """Доминирующая частота после декодирования — тот же тон (проверка ресемплинга)."""
    audio = load_audio_from_file(encoded[name])
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    peak_hz = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[int(np.argmax(spectrum))]
    assert abs(peak_hz - TONE_HZ) < 10, f"{name}: пик на {peak_hz:.1f} Гц вместо {TONE_HZ}"


def test_truncated_input_raises(encoded):
    """Битый/обрезанный файл даёт исключение, а не пустой массив.

    Регрессия: `ffmpeg -i pipe:0` на mp4 возвращал rc=0 и 0 сэмплов, ошибка
    всплывала позже как загадочное "Audio data is empty".
    """
    with pytest.raises(Exception):
        load_audio_from_file(encoded["chrome.webm"][:200])


def test_decode_is_fast(encoded):
    """Декодирование не должно стоить как fork ffmpeg (~34ms только на запуск)."""
    data = encoded["chrome.webm"]
    load_audio_from_file(data)  # warmup
    start = time.perf_counter()
    for _ in range(5):
        load_audio_from_file(data)
    per_call_ms = (time.perf_counter() - start) / 5 * 1000
    assert per_call_ms < 30, f"декодирование 3s webm заняло {per_call_ms:.1f}ms"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
