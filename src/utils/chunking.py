"""
OAITT — Open AI Transformer Transcriber.

Smart audio chunking utilities.
Provides functions for splitting audio into chunks with intelligent
boundary detection (silence-based splitting) for better transcription quality.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from typing import List, Tuple, Optional, Generator
import numpy as np

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def find_silence_points(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.3,
) -> List[int]:
    """
    Find silence points in audio that are suitable for splitting.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        silence_threshold: Amplitude threshold below which audio is considered silence.
        min_silence_duration: Minimum duration of silence in seconds.

    Returns:
        List of sample indices where silence occurs (midpoints of silence regions).
    """
    min_silence_samples = int(min_silence_duration * sample_rate)

    # Compute absolute amplitude
    amplitude = np.abs(audio)

    # Find regions below threshold
    is_silent = amplitude < silence_threshold

    # Find contiguous silent regions
    silence_points = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            silence_start = i
        elif not silent and in_silence:
            in_silence = False
            silence_length = i - silence_start
            if silence_length >= min_silence_samples:
                # Use midpoint of silence region
                midpoint = silence_start + silence_length // 2
                silence_points.append(midpoint)

    # Handle case where audio ends in silence
    if in_silence:
        silence_length = len(audio) - silence_start
        if silence_length >= min_silence_samples:
            midpoint = silence_start + silence_length // 2
            silence_points.append(midpoint)

    return silence_points


def find_best_split_point(
    audio: np.ndarray,
    target_sample: int,
    search_window: int,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.1,
) -> int:
    """
    Find the best split point near a target sample position.

    Searches for silence or low-amplitude regions near the target position
    to avoid cutting in the middle of speech.

    Args:
        audio: Audio data as numpy array.
        target_sample: Target sample position for splitting.
        search_window: Number of samples to search around target (on each side).
        sample_rate: Sample rate of the audio.
        silence_threshold: Amplitude threshold for silence detection.
        min_silence_duration: Minimum silence duration in seconds.

    Returns:
        Best sample index for splitting (closest silence to target, or target if none found).
    """
    start = max(0, target_sample - search_window)
    end = min(len(audio), target_sample + search_window)

    # Extract the search region
    search_region = audio[start:end]

    # Find silence points in the search region
    silence_points = find_silence_points(
        search_region,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        min_silence_duration=min_silence_duration,
    )

    if silence_points:
        # Convert to absolute positions
        absolute_points = [start + p for p in silence_points]
        # Find closest to target
        best_point = min(absolute_points, key=lambda x: abs(x - target_sample))
        return best_point

    # Fallback: find the minimum amplitude point in the window
    if len(search_region) > 0:
        amplitude = np.abs(search_region)
        # Use a small window average to find smooth low points
        window_size = min(int(0.05 * sample_rate), len(amplitude))  # 50ms window
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(amplitude, kernel, mode='same')
            min_idx = np.argmin(smoothed)
        else:
            min_idx = np.argmin(amplitude)
        return start + min_idx

    return target_sample


def split_audio_smart(
    audio: np.ndarray,
    chunk_duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    min_chunk_duration_sec: float = 1.0,
    overlap_sec: float = 0.0,
    search_window_sec: float = 2.0,
    silence_threshold: float = 0.01,
) -> Generator[Tuple[np.ndarray, float, float], None, None]:
    """
    Split audio into chunks with smart boundary detection.

    Attempts to split at silence points or low-amplitude regions
    to avoid cutting in the middle of words/speech.

    Args:
        audio: Audio data as numpy array.
        chunk_duration_sec: Target duration for each chunk in seconds.
        sample_rate: Sample rate of the audio.
        min_chunk_duration_sec: Minimum chunk duration in seconds.
        overlap_sec: Overlap between chunks in seconds (for context).
        search_window_sec: Window size in seconds to search for split points.
        silence_threshold: Amplitude threshold for silence detection.

    Yields:
        Tuples of (chunk_audio, start_time_sec, end_time_sec).
    """
    total_samples = len(audio)
    total_duration = total_samples / sample_rate

    if total_duration <= chunk_duration_sec:
        # Audio is short enough, return as single chunk
        yield audio, 0.0, total_duration
        return

    chunk_samples = int(chunk_duration_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    search_window_samples = int(search_window_sec * sample_rate)
    min_chunk_samples = int(min_chunk_duration_sec * sample_rate)

    pos = 0
    chunk_idx = 0

    while pos < total_samples:
        # Target end position
        target_end = pos + chunk_samples

        if target_end >= total_samples:
            # Last chunk - take everything remaining
            chunk = audio[pos:]
            start_sec = pos / sample_rate
            end_sec = total_samples / sample_rate
            if len(chunk) >= min_chunk_samples:
                yield chunk, start_sec, end_sec
            break

        # Find best split point
        actual_end = find_best_split_point(
            audio,
            target_end,
            search_window_samples,
            sample_rate=sample_rate,
            silence_threshold=silence_threshold,
        )

        # Ensure minimum chunk size
        if actual_end - pos < min_chunk_samples:
            actual_end = min(pos + chunk_samples, total_samples)

        chunk = audio[pos:actual_end]
        start_sec = pos / sample_rate
        end_sec = actual_end / sample_rate

        yield chunk, start_sec, end_sec

        # Move position, accounting for overlap
        pos = max(pos + min_chunk_samples, actual_end - overlap_samples)
        chunk_idx += 1

    logger.debug(f"Split audio into {chunk_idx + 1} chunks using smart boundaries")


def split_audio_fixed(
    audio: np.ndarray,
    chunk_duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    overlap_sec: float = 0.0,
) -> Generator[Tuple[np.ndarray, float, float], None, None]:
    """
    Split audio into fixed-size chunks (simple splitting without silence detection).

    Args:
        audio: Audio data as numpy array.
        chunk_duration_sec: Duration for each chunk in seconds.
        sample_rate: Sample rate of the audio.
        overlap_sec: Overlap between chunks in seconds.

    Yields:
        Tuples of (chunk_audio, start_time_sec, end_time_sec).
    """
    total_samples = len(audio)
    total_duration = total_samples / sample_rate

    if total_duration <= chunk_duration_sec:
        yield audio, 0.0, total_duration
        return

    chunk_samples = int(chunk_duration_sec * sample_rate)
    step_samples = chunk_samples - int(overlap_sec * sample_rate)

    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio[pos:end]
        start_sec = pos / sample_rate
        end_sec = end / sample_rate
        yield chunk, start_sec, end_sec
        pos += step_samples


def estimate_optimal_chunk_size(
    audio_duration_sec: float,
    max_chunk_sec: float = 30.0,
    min_chunk_sec: float = 5.0,
    target_chunks: int = 10,
) -> float:
    """
    Estimate optimal chunk size based on audio duration.

    For shorter audio, uses larger chunks (fewer splits).
    For longer audio, may use smaller chunks for better parallelization.

    Args:
        audio_duration_sec: Total audio duration in seconds.
        max_chunk_sec: Maximum chunk duration.
        min_chunk_sec: Minimum chunk duration.
        target_chunks: Target number of chunks for long audio.

    Returns:
        Recommended chunk duration in seconds.
    """
    if audio_duration_sec <= max_chunk_sec:
        return audio_duration_sec

    # Calculate chunk size to get approximately target_chunks
    ideal_chunk = audio_duration_sec / target_chunks

    # Clamp to min/max
    return max(min_chunk_sec, min(max_chunk_sec, ideal_chunk))


class AudioChunker:
    """
    Audio chunker with configurable splitting strategy.

    Supports both smart (silence-based) and fixed splitting modes.
    Can be used as a context manager or directly.

    Example:
        >>> chunker = AudioChunker(chunk_duration_sec=30.0, use_smart_split=True)
        >>> for chunk, start, end in chunker.split(audio):
        ...     result = model.transcribe(chunk)
        ...     print(f"[{start:.2f}-{end:.2f}]: {result}")
    """

    def __init__(
        self,
        chunk_duration_sec: float = 30.0,
        min_chunk_duration_sec: float = 5.0,
        overlap_sec: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
        use_smart_split: bool = True,
        silence_threshold: float = 0.01,
        search_window_sec: float = 2.0,
    ):
        """
        Initialize the audio chunker.

        Args:
            chunk_duration_sec: Target chunk duration in seconds.
            min_chunk_duration_sec: Minimum chunk duration in seconds.
            overlap_sec: Overlap between consecutive chunks.
            sample_rate: Audio sample rate.
            use_smart_split: Whether to use silence-based smart splitting.
            silence_threshold: Amplitude threshold for silence detection.
            search_window_sec: Search window for finding split points.
        """
        self.chunk_duration_sec = chunk_duration_sec
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.sample_rate = sample_rate
        self.use_smart_split = use_smart_split
        self.silence_threshold = silence_threshold
        self.search_window_sec = search_window_sec

    def split(
        self, audio: np.ndarray
    ) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """
        Split audio into chunks.

        Args:
            audio: Audio data as numpy array.

        Yields:
            Tuples of (chunk_audio, start_time_sec, end_time_sec).
        """
        if self.use_smart_split:
            yield from split_audio_smart(
                audio,
                chunk_duration_sec=self.chunk_duration_sec,
                sample_rate=self.sample_rate,
                min_chunk_duration_sec=self.min_chunk_duration_sec,
                overlap_sec=self.overlap_sec,
                search_window_sec=self.search_window_sec,
                silence_threshold=self.silence_threshold,
            )
        else:
            yield from split_audio_fixed(
                audio,
                chunk_duration_sec=self.chunk_duration_sec,
                sample_rate=self.sample_rate,
                overlap_sec=self.overlap_sec,
            )

    def get_chunks_list(
        self, audio: np.ndarray
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio and return all chunks as a list.

        Args:
            audio: Audio data as numpy array.

        Returns:
            List of tuples (chunk_audio, start_time_sec, end_time_sec).
        """
        return list(self.split(audio))

    def __repr__(self) -> str:
        return (
            f"AudioChunker(chunk_duration_sec={self.chunk_duration_sec}, "
            f"use_smart_split={self.use_smart_split})"
        )
