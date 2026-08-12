#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Проверка параллельной транскрипции на MLX-движках (in-process, без HTTP).

Что проверяется:
  - N потоков одновременно вызывают transcribe() на ОДНОМ экземпляре модели
    (режим GIGAAM_MLX_LOCK_FREE=true) — не падает, результаты не портятся;
  - параллельный результат посимвольно совпадает с последовательным;
  - wall-clock параллельного прогона не хуже последовательного (нет полной
    сериализации на локе).

HTTP-уровень покрыт tests/test_parallel_benchmark.py — этот тест ловит гонки
внутри самого движка, где бенчмарк их не увидит.

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python -m pytest tests/test_mlx_parallel.py -v -s
    GIGAAM_MLX_MODEL_TYPE=rnnt python -m pytest tests/test_mlx_parallel.py -v -s
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from src.config import SAMPLE_RATE
from src.utils.audio import load_audio_from_path

THREADS = int(os.environ.get("TEST_PARALLEL_THREADS", "4"))
CLIP_SEC = float(os.environ.get("TEST_CLIP_SEC", "10"))

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="MLX-движок работает только на Apple Silicon"
)


def _sample_path() -> Path:
    for d in ("sample-data", "samples"):
        for f in sorted(Path(d).glob("*")):
            if f.suffix.lower() in (".ogg", ".wav", ".mp3", ".webm", ".m4a"):
                return f
    pytest.skip("нет аудио в sample-data/ или samples/")


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    data = load_audio_from_path(str(_sample_path()))
    return data[: int(CLIP_SEC * SAMPLE_RATE)]


@pytest.fixture(scope="module")
def model():
    """Один экземпляр MLX-модели, общий для всех потоков — как в проде."""
    # vendor/gigaam-mlx кладётся в PYTHONPATH скриптами run_gigaam_mlx_*.sh
    vendor = Path("vendor/gigaam-mlx").resolve()
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    from src.asr.gigaam_mlx import GIGAAM_MLX_LOCK_FREE, GigaAMMLXASR

    m = GigaAMMLXASR()
    try:
        m.ensure_model_loaded()
    except Exception as e:
        pytest.skip(f"MLX модель недоступна: {e}")

    print(f"\nМодель: {m.model_type}, lock_free={GIGAAM_MLX_LOCK_FREE}, потоков={THREADS}")
    yield m
    m.release_model()


def _transcribe(model, audio) -> str:
    return model.transcribe(
        audio=audio,
        task="transcribe",
        language="ru",
        word_timestamps=False,
        output="text",
    )


def test_parallel_matches_sequential(model, audio):
    """Параллельные вызовы дают тот же текст, что и последовательные, и не медленнее."""
    baseline = _transcribe(model, audio)  # прогрев + эталон
    assert baseline.strip(), "эталонная транскрипция пустая"

    seq_start = time.perf_counter()
    for _ in range(THREADS):
        _transcribe(model, audio)
    seq_elapsed = time.perf_counter() - seq_start

    par_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(lambda _: _transcribe(model, audio), range(THREADS)))
    par_elapsed = time.perf_counter() - par_start

    print(
        f"seq {THREADS}x: {seq_elapsed:.2f}s | par {THREADS}x: {par_elapsed:.2f}s "
        f"| speedup {seq_elapsed / par_elapsed:.2f}x"
    )

    for i, text in enumerate(results):
        assert text == baseline, (
            f"поток {i} вернул другой текст (гонка в движке):\n"
            f"  ожидалось: {baseline[:120]}\n"
            f"  получено:  {text[:120]}"
        )

    # Параллельный прогон не должен быть МЕДЛЕННЕЕ последовательного.
    # Порог мягкий: GPU-исполнение всё равно сериализуется runtime'ом,
    # выигрыш идёт только с CPU-части (мел-спектр, чанкование, декод токенов).
    assert par_elapsed < seq_elapsed * 1.2, (
        f"параллельный прогон медленнее последовательного: "
        f"{par_elapsed:.2f}s против {seq_elapsed:.2f}s"
    )


def test_parallel_mixed_lengths(model, audio):
    """Разные длины аудио одновременно — проверка shape-кеша/паддинга под гонкой."""
    clips = [audio[: int(sec * SAMPLE_RATE)] for sec in (2, 5, 8, 10)][:THREADS]
    expected = [_transcribe(model, c) for c in clips]

    with ThreadPoolExecutor(max_workers=len(clips)) as pool:
        got = list(pool.map(lambda c: _transcribe(model, c), clips))

    for i, (want, have) in enumerate(zip(expected, got)):
        assert want == have, f"клип {i}: результат разошёлся под нагрузкой"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
