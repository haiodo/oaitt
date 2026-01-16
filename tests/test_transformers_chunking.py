#!/usr/bin/env python3
"""
Tests for Transformers chunking behavior — ensure tiny final remainder is merged
into the previous chunk (to avoid creating extremely short chunks that break
feature extraction), and that sufficiently large remainders are kept as their
own chunk.
"""

import numpy as np

from src.asr.transformers import TransformersASR
from src.config import SAMPLE_RATE, GIGAAM_CHUNK_SEC, GIGAAM_MIN_CHUNK_SEC


def test_merge_small_final_remainder(monkeypatch):
    """
    Create an audio buffer whose length results in a tiny final remainder
    (e.g. chunk_samples + 69). The final tiny remainder should be merged
    into the previous chunk, so only one chunk should be processed by pipeline.
    """
    model = TransformersASR()

    chunk_samples = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
    audio_len = chunk_samples + 69
    audio = np.zeros(audio_len, dtype=np.float32)

    called = []

    def fake_pipeline(audio_input, generate_kwargs=None, return_timestamps=None):
        called.append(len(audio_input))
        # Return a pipeline-like dict with 'text'
        return {"text": "ok"}

    monkeypatch.setattr(model, "pipeline", fake_pipeline)

    results = model._transcribe_chunked_pipeline(audio, {})

    # Expect a single pipeline call covering the entire audio (merged remainder)
    assert len(called) == 1
    assert called[0] == audio_len

    # Result should contain one entry with correct timing
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["text"] == "ok"
    assert results[0]["start"] == 0.0
    assert abs(results[0]["end"] - (audio_len / SAMPLE_RATE)) < 1e-6


def test_keep_large_remainder(monkeypatch):
    """
    Create an audio buffer where the remainder after the last full chunk is
    larger than the minimum chunk size. In this case we should get two chunks:
    the full chunk and the remainder chunk.
    """
    model = TransformersASR()

    chunk_samples = int(GIGAAM_CHUNK_SEC * SAMPLE_RATE)
    min_samples = int(GIGAAM_MIN_CHUNK_SEC * SAMPLE_RATE)
    remainder = min_samples + 1000

    audio_len = chunk_samples + remainder
    audio = np.zeros(audio_len, dtype=np.float32)

    called = []

    def fake_pipeline(audio_input, generate_kwargs=None, return_timestamps=None):
        called.append(len(audio_input))
        return {"text": "ok"}

    monkeypatch.setattr(model, "pipeline", fake_pipeline)

    results = model._transcribe_chunked_pipeline(audio, {})

    # Expect two pipeline calls: first full chunk, then remainder chunk
    assert len(called) == 2
    assert called[0] == chunk_samples
    assert called[1] == remainder

    # Results should contain two segments with correct start/end times
    assert len(results) == 2
    assert results[0]["start"] == 0.0
    assert results[0]["end"] == chunk_samples / SAMPLE_RATE
    assert results[1]["start"] == chunk_samples / SAMPLE_RATE
    assert results[1]["end"] == audio_len / SAMPLE_RATE
