#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Тесты транскрипции через OpenAI-совместимый API.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import os
import time
import subprocess
import signal
import sys
import tempfile
import requests
from pathlib import Path

import soundfile as sf
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Sample data directory for test files
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample-data"

# Test audio file
TEST_AUDIO_FILE = "Sobolev_Andrey_1_0_00-2_17.ogg"

# Server configuration
SERVER_HOST = "localhost"
SERVER_PORT = 9007
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
API_ENDPOINT = f"{SERVER_URL}/v1/audio/transcriptions"

# Default API token
API_TOKEN = "key"


def get_test_audio() -> Path:
    """Get the test audio file from sample-data directory."""
    audio_path = SAMPLE_DATA_DIR / TEST_AUDIO_FILE
    if not audio_path.exists():
        raise FileNotFoundError(f"Test audio file not found: {audio_path}")
    return audio_path


def extract_audio_segment(input_path: Path, duration_sec: float = 29.0) -> Path:
    """
    Extract a segment from audio file.

    Args:
        input_path: Path to input audio file
        duration_sec: Duration to extract in seconds

    Returns:
        Path to temporary file with extracted audio
    """
    # Read audio file
    audio_data, sample_rate = sf.read(input_path)

    # Calculate number of samples for the duration
    samples_needed = int(duration_sec * sample_rate)

    # Extract segment (from beginning)
    if len(audio_data.shape) > 1:
        # Stereo to mono
        audio_data = audio_data.mean(axis=1)

    # Limit to requested duration
    audio_segment = audio_data[:samples_needed]

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_segment, sample_rate)

    return Path(temp_file.name)


def wait_for_server(timeout: float = 120.0, check_interval: float = 2.0) -> bool:
    """
    Wait for the server to become available.

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if server is available, False if timeout reached
    """
    start_time = time.time()
    health_url = f"{SERVER_URL}/health"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(check_interval)

    return False


def transcribe_audio(audio_path: Path, language: str = "ru") -> dict:
    """
    Send audio file to transcription API.

    Args:
        audio_path: Path to audio file
        language: Language code

    Returns:
        API response as dictionary
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
    }

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {
            "model": "whisper-1",
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }

        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=300,  # 5 minutes timeout
        )

    response.raise_for_status()
    return response.json()


def test_transcription_with_running_server():
    """
    Test transcription with an already running server.

    This test assumes the server is already running.
    Use this test for quick validation.
    """
    # Check if server is available
    if not wait_for_server(timeout=5.0):
        print("Server is not running. Please start it manually.")
        return

    # Get test audio file
    sample_path = get_test_audio()
    print(f"Using sample: {sample_path}")

    # Extract 29 seconds
    audio_path = extract_audio_segment(sample_path, duration_sec=29.0)
    print(f"Extracted segment: {audio_path}")

    try:
        # Transcribe
        start_time = time.time()
        result = transcribe_audio(audio_path)
        elapsed = time.time() - start_time

        print(f"\nTranscription completed in {elapsed:.2f}s")
        print(f"Text: {result.get('text', '')[:200]}...")
        print(f"Language: {result.get('language')}")
        print(f"Duration: {result.get('duration')}s")

        # Verify result
        assert "text" in result, "Response should contain 'text' field"
        assert len(result["text"]) > 0, "Transcription should not be empty"

        print("\n✓ Test passed!")

    finally:
        # Cleanup temp file
        if audio_path.exists():
            audio_path.unlink()


def run_test():
    """Main test runner."""
    print("=" * 60)
    print("OAITT Transcription Test")
    print("=" * 60)

    test_transcription_with_running_server()


if __name__ == "__main__":
    run_test()
