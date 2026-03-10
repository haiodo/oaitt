#!/usr/bin/env python3
"""
Tests for memory leak detection in GigaAM ASR.

These tests verify that memory usage doesn't grow unbounded
when performing multiple transcriptions.
"""

import gc
import os
import sys

import numpy as np
import pytest

# Skip tests if not running with gigaam engine
if os.environ.get("ASR_ENGINE") != "gigaam":
    pytest.skip("Skipping GigaAM memory tests - ASR_ENGINE is not gigaam", allow_module_level=True)

try:
    import torch
    from src.asr.gigaam import GigaAMASR
    from src.config import SAMPLE_RATE
    from src.utils.device import clear_memory_cache, get_process_memory_mb, get_gpu_memory_mb
    GIGAAM_AVAILABLE = True
except ImportError as e:
    GIGAAM_AVAILABLE = False
    pytest.skip(f"GigaAM not available: {e}", allow_module_level=True)


class TestGigaAMMemoryLeaks:
    """Test suite for detecting memory leaks in GigaAM ASR."""

    @pytest.fixture(scope="class")
    def asr_model(self):
        """Fixture to create and load GigaAM model."""
        model = GigaAMASR()
        model.load_model()
        yield model
        # Cleanup after tests
        model.release_model()
        clear_memory_cache()

    def _create_test_audio(self, duration_sec: float = 5.0) -> np.ndarray:
        """Create test audio data (silence)."""
        samples = int(duration_sec * SAMPLE_RATE)
        return np.zeros(samples, dtype=np.float32)

    def _get_memory_usage(self) -> dict:
        """Get current memory usage."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return {
            "process_mb": get_process_memory_mb(),
            "gpu_mb": get_gpu_memory_mb() or 0,
        }

    def test_single_transcription_no_leak(self, asr_model):
        """Test that a single transcription doesn't leave memory unreleased."""
        audio = self._create_test_audio(duration_sec=3.0)
        
        # Record memory before
        mem_before = self._get_memory_usage()
        
        # Perform transcription
        result = asr_model.transcribe(
            audio=audio,
            task="transcribe",
            language="ru",
            word_timestamps=False,
            output="json",
        )
        
        # Record memory after
        mem_after = self._get_memory_usage()
        
        # Allow some tolerance for memory fluctuation (10% or 50MB)
        process_increase = mem_after["process_mb"] - mem_before["process_mb"]
        gpu_increase = mem_after["gpu_mb"] - mem_before["gpu_mb"]
        
        assert process_increase < 100, (
            f"Process memory increased by {process_increase:.1f}MB after single transcription. "
            "Possible memory leak."
        )
        
        # GPU memory should not grow significantly for single transcription
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            assert gpu_increase < 200, (
                f"GPU memory increased by {gpu_increase:.1f}MB after single transcription. "
                "Possible memory leak."
            )

    def test_multiple_transcriptions_memory_stable(self, asr_model):
        """Test that memory doesn't grow unbounded over multiple transcriptions."""
        audio = self._create_test_audio(duration_sec=2.0)
        
        # Warm up
        for _ in range(3):
            asr_model.transcribe(
                audio=audio,
                task="transcribe",
                language="ru",
                word_timestamps=False,
                output="json",
            )
        
        # Record memory after warm-up
        mem_baseline = self._get_memory_usage()
        
        # Perform multiple transcriptions
        num_iterations = 10
        for i in range(num_iterations):
            asr_model.transcribe(
                audio=audio,
                task="transcribe",
                language="ru",
                word_timestamps=False,
                output="json",
            )
        
        # Record memory after iterations
        mem_after = self._get_memory_usage()
        
        # Calculate memory growth
        process_growth = mem_after["process_mb"] - mem_baseline["process_mb"]
        gpu_growth = mem_after["gpu_mb"] - mem_baseline["gpu_mb"]
        
        # Memory should not grow more than 50MB per iteration on average
        avg_process_growth_per_iter = process_growth / num_iterations
        avg_gpu_growth_per_iter = gpu_growth / num_iterations
        
        assert avg_process_growth_per_iter < 10, (
            f"Process memory growing by {avg_process_growth_per_iter:.1f}MB per iteration. "
            f"Total growth: {process_growth:.1f}MB over {num_iterations} iterations. "
            "Possible memory leak."
        )
        
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            assert avg_gpu_growth_per_iter < 20, (
                f"GPU memory growing by {avg_gpu_growth_per_iter:.1f}MB per iteration. "
                f"Total growth: {gpu_growth:.1f}MB over {num_iterations} iterations. "
                "Possible memory leak."
            )

    def test_chunked_transcription_no_leak(self, asr_model):
        """Test that chunked transcription for long audio doesn't leak memory."""
        # Create longer audio that will trigger chunking (45 seconds)
        audio = self._create_test_audio(duration_sec=45.0)
        
        # Record memory before
        mem_before = self._get_memory_usage()
        
        # Perform transcription (should use chunked processing)
        result = asr_model.transcribe(
            audio=audio,
            task="transcribe",
            language="ru",
            word_timestamps=False,
            output="json",
        )
        
        # Record memory after
        mem_after = self._get_memory_usage()
        
        # Allow more tolerance for chunked processing
        process_increase = mem_after["process_mb"] - mem_before["process_mb"]
        
        assert process_increase < 200, (
            f"Process memory increased by {process_increase:.1f}MB after chunked transcription. "
            "Possible memory leak in chunking logic."
        )

    def test_tensor_transcription_clears_memory(self, asr_model):
        """Test that direct tensor transcription properly releases memory."""
        import torch
        
        audio = self._create_test_audio(duration_sec=3.0)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Perform tensor transcription directly
        result = asr_model._transcribe_audio_tensor(audio)
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Verify we got a result
        assert isinstance(result, str)
        
        # If CUDA is available, check that peak memory is reasonable
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            # Peak memory should be less than 2GB for 3 second audio
            assert peak_mb < 2048, (
                f"Peak GPU memory usage was {peak_mb:.1f}MB for 3-second audio. "
                "This seems excessive."
            )


class TestGigaAMModelLifecycle:
    """Test proper resource cleanup during model lifecycle."""

    def test_model_release_frees_memory(self):
        """Test that releasing model actually frees memory."""
        if not GIGAAM_AVAILABLE:
            pytest.skip("GigaAM not available")
        
        model = GigaAMASR()
        
        # Record baseline memory
        gc.collect()
        clear_memory_cache()
        mem_before_load = get_process_memory_mb()
        
        # Load model
        model.load_model()
        mem_after_load = get_process_memory_mb()
        
        # Memory should increase after loading
        memory_increase = mem_after_load - mem_before_load
        assert memory_increase > 50, "Model should use significant memory after loading"
        
        # Release model
        model.release_model()
        gc.collect()
        clear_memory_cache()
        mem_after_release = get_process_memory_mb()
        
        # Memory should decrease significantly (allow 20% tolerance)
        memory_freed = mem_after_load - mem_after_release
        assert memory_freed > memory_increase * 0.5, (
            f"Only {memory_freed:.1f}MB freed out of {memory_increase:.1f}MB used. "
            "Model release may not be cleaning up properly."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
