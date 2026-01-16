#!/usr/bin/env python3
"""
Tests for GigaAM chunking behavior — ensure tiny final remainder is merged into
the previous chunk (to avoid creating extremely short chunks that break
feature extraction), and that sufficiently large remainders are kept as their
own chunk.
"""

import numpy as np

from src.asr.gigaam import GigaAMASR
from src.config import SAMPLE_RATE, GIGAAM_CHUNK_SEC, GIGAAM_MIN_CHUNK_SEC


def test_merge_small_final_remainder(monkeypatch):
    """
    Create an audio buffer whose length results in a tiny final remainder
    (e.g. chunk_samples + 69). The final tiny remainder should be merged
    into the previous chunk, so only one chunk should be transcribed.
    """
    model = GigaAMASR()

    chunk_samples = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
    audio_len = chunk_samples + 69
    audio = np.zeros(audio_len, dtype=np.float32)

    called = []

    def fake_transcribe(chunk_audio, start_sec, max_chunk_sec, min_chunk_sec):
        called.append((start_sec, len(chunk_audio)))
        duration = len(chunk_audio) / SAMPLE_RATE
        return [{"text": "ok", "boundaries": (start_sec, start_sec + duration)}]

    monkeypatch.setattr(model, "_transcribe_chunk_with_retry", fake_transcribe)

    results = model._transcribe_chunked(audio)

    # Expect a single call (merged remainder) covering the entire audio
    assert len(called) == 1
    assert called[0][0] == 0.0
    assert called[0][1] == audio_len

    assert isinstance(results, list)
    assert results and results[0]["transcription"] == "ok"


def test_keep_large_remainder(monkeypatch):
    """
    Create an audio buffer where the remainder after the last full chunk is
    larger than the minimum chunk size. In this case we should get two chunks:
    the full chunk and the remainder chunk.
    """
    model = GigaAMASR()

    chunk_samples = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
    min_samples = int(GIGAAM_MIN_CHUNK_SEC * SAMPLE_RATE)
    remainder = min_samples + 1000

    audio_len = chunk_samples + remainder
    audio = np.zeros(audio_len, dtype=np.float32)

    called = []

    def fake_transcribe(chunk_audio, start_sec, max_chunk_sec, min_chunk_sec):
        called.append((start_sec, len(chunk_audio)))
        duration = len(chunk_audio) / SAMPLE_RATE
        return [{"text": "ok", "boundaries": (start_sec, start_sec + duration)}]

    monkeypatch.setattr(model, "_transcribe_chunk_with_retry", fake_transcribe)

    results = model._transcribe_chunked(audio)

    # Expect two calls: first for the full chunk, second for the remainder
    assert len(called) == 2
    assert called[0][0] == 0.0
    assert called[0][1] == chunk_samples
    assert called[1][0] == chunk_samples / SAMPLE_RATE
    assert called[1][1] == remainder

    assert results and all("transcription" in r for r in results)
