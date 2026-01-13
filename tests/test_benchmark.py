#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Бенчмарк тест для сравнения производительности различных ASR движков.

Запускает последовательно:
1. GigaAM (run_gigaam.sh)
2. Whisper Large V3 через Transformers (run_whisper_large_v3.sh)
3. WhisperX Large V3 (run_whisperx_large_v3.sh)

Для каждого движка замеряется время транскрипции и результаты
выводятся в сравнительную таблицу.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python -m tests.test_benchmark
"""

import os
import sys
import time
import signal
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import soundfile as sf
import requests

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
HEALTH_ENDPOINT = f"{SERVER_URL}/health"

# Default API token
API_TOKEN = "key"

# Test audio duration in seconds
TEST_AUDIO_DURATION = 29.0

# Server startup timeout in seconds
SERVER_STARTUP_TIMEOUT = 180.0

# Scripts to benchmark
BENCHMARK_SCRIPTS = [
    ("GigaAM", "run_gigaam.sh"),
    ("Whisper Large V3 (Transformers)", "run_whisper_large_v3.sh"),
    ("WhisperX Large V3", "run_whisperx_large_v3.sh"),
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    engine_name: str
    script_name: str
    transcription_time: float
    text: str
    text_length: int
    audio_duration: float
    speed_ratio: float  # audio_duration / transcription_time
    success: bool
    error: Optional[str] = None


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

    actual_duration = len(audio_segment) / sample_rate

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_segment, sample_rate)

    return Path(temp_file.name), actual_duration


def wait_for_server(timeout: float = SERVER_STARTUP_TIMEOUT, check_interval: float = 2.0) -> bool:
    """
    Wait for the server to become available.

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        True if server is available, False if timeout reached
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(check_interval)

    return False


def stop_server(process: subprocess.Popen) -> None:
    """
    Stop the server process.

    Args:
        process: Server subprocess
    """
    if process is None:
        return

    try:
        # Send SIGTERM
        process.terminate()

        # Wait for graceful shutdown
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Force kill if not responding
            process.kill()
            process.wait(timeout=5)
    except Exception as e:
        print(f"Error stopping server: {e}")


def start_server(script_name: str) -> Optional[subprocess.Popen]:
    """
    Start the server with the specified script.

    Args:
        script_name: Name of the script to run (e.g., "run_gigaam.sh")

    Returns:
        Server subprocess or None if failed
    """
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return None

    # Start server process
    process = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for clean termination
    )

    return process


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
        }

        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=600,  # 10 minutes timeout for slow models
        )

    response.raise_for_status()
    return response.json()


def run_benchmark(
    engine_name: str,
    script_name: str,
    audio_path: Path,
    audio_duration: float,
) -> BenchmarkResult:
    """
    Run benchmark for a single ASR engine.

    Args:
        engine_name: Human-readable engine name
        script_name: Script filename
        audio_path: Path to test audio file
        audio_duration: Duration of audio in seconds

    Returns:
        BenchmarkResult with timing and transcription data
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {engine_name}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")

    process = None

    try:
        # Start server
        print("Starting server...")
        process = start_server(script_name)

        if process is None:
            return BenchmarkResult(
                engine_name=engine_name,
                script_name=script_name,
                transcription_time=0,
                text="",
                text_length=0,
                audio_duration=audio_duration,
                speed_ratio=0,
                success=False,
                error="Failed to start server",
            )

        # Wait for server to be ready
        print(f"Waiting for server (up to {SERVER_STARTUP_TIMEOUT}s)...")
        if not wait_for_server():
            return BenchmarkResult(
                engine_name=engine_name,
                script_name=script_name,
                transcription_time=0,
                text="",
                text_length=0,
                audio_duration=audio_duration,
                speed_ratio=0,
                success=False,
                error="Server startup timeout",
            )

        print("Server is ready!")

        # Run transcription
        print(f"Transcribing {audio_duration:.1f}s audio...")
        start_time = time.time()
        result = transcribe_audio(audio_path)
        transcription_time = time.time() - start_time

        text = result.get("text", "")
        speed_ratio = audio_duration / transcription_time if transcription_time > 0 else 0

        print(f"Transcription completed in {transcription_time:.2f}s")
        print(f"Speed: {speed_ratio:.2f}x realtime")
        print(f"Text length: {len(text)} chars")
        print(f"Preview: {text[:100]}...")

        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=transcription_time,
            text=text,
            text_length=len(text),
            audio_duration=audio_duration,
            speed_ratio=speed_ratio,
            success=True,
        )

    except requests.exceptions.RequestException as e:
        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=0,
            text="",
            text_length=0,
            audio_duration=audio_duration,
            speed_ratio=0,
            success=False,
            error=f"Request error: {e}",
        )

    except Exception as e:
        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=0,
            text="",
            text_length=0,
            audio_duration=audio_duration,
            speed_ratio=0,
            success=False,
            error=str(e),
        )

    finally:
        # Stop server
        if process is not None:
            print("Stopping server...")
            # Kill process group
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Give some time for port to be released
            time.sleep(3)


def print_results_table(results: list[BenchmarkResult]) -> None:
    """
    Print benchmark results as a formatted table.

    Args:
        results: List of benchmark results
    """
    print("\n")
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Engine':<40} {'Time (s)':<12} {'Speed':<12} {'Status':<12}")
    print("-" * 80)

    # Results
    for r in results:
        if r.success:
            status = "✓ OK"
            time_str = f"{r.transcription_time:.2f}"
            speed_str = f"{r.speed_ratio:.2f}x"
        else:
            status = f"✗ {r.error[:20]}" if r.error else "✗ Failed"
            time_str = "N/A"
            speed_str = "N/A"

        print(f"{r.engine_name:<40} {time_str:<12} {speed_str:<12} {status:<12}")

    print("-" * 80)

    # Summary
    successful = [r for r in results if r.success]
    if successful:
        fastest = min(successful, key=lambda r: r.transcription_time)
        print(f"\nFastest: {fastest.engine_name} ({fastest.transcription_time:.2f}s, {fastest.speed_ratio:.2f}x realtime)")

    print("\n")


def main():
    """Main benchmark runner."""
    print("=" * 80)
    print("OAITT ASR Engine Benchmark")
    print("=" * 80)

    # Find and prepare audio
    print("\nPreparing test audio...")
    sample_path = get_test_audio()
    print(f"Using sample: {sample_path.name}")

    audio_path, audio_duration = extract_audio_segment(sample_path, TEST_AUDIO_DURATION)
    print(f"Extracted {audio_duration:.1f}s audio segment: {audio_path}")

    results = []

    try:
        # Run benchmarks for each engine
        for engine_name, script_name in BENCHMARK_SCRIPTS:
            # Check if script exists
            script_path = PROJECT_ROOT / script_name
            if not script_path.exists():
                print(f"\nSkipping {engine_name}: script {script_name} not found")
                results.append(BenchmarkResult(
                    engine_name=engine_name,
                    script_name=script_name,
                    transcription_time=0,
                    text="",
                    text_length=0,
                    audio_duration=audio_duration,
                    speed_ratio=0,
                    success=False,
                    error="Script not found",
                ))
                continue

            result = run_benchmark(engine_name, script_name, audio_path, audio_duration)
            results.append(result)

        # Print results table
        print_results_table(results)

    finally:
        # Cleanup temp file
        if audio_path.exists():
            audio_path.unlink()
            print(f"Cleaned up temporary file: {audio_path}")


if __name__ == "__main__":
    main()
