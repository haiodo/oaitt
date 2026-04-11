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
    python -m tests.test_benchmark [--mode short|long|full] [--iterations N]

    Modes:
        short (default): 20s audio, 5 iterations - quick benchmark
        long:  60s audio, 3 iterations - tests chunked transcription
        full:  full file (~137s), 1 iteration - complete longform test
"""

import argparse
import os
import sys
import time
import signal
import socket
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

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

# Test modes configuration
TEST_MODES = {
    "short": {
        "duration": 20.0,      # 20 seconds - within GigaAM short audio limit
        "iterations": 5,
        "description": "Quick benchmark with short audio",
    },
    "long": {
        "duration": 60.0,      # 60 seconds - tests chunked transcription
        "iterations": 3,
        "description": "Chunked transcription test",
    },
    "full": {
        "duration": None,      # None = use full file
        "iterations": 1,
        "description": "Full file longform test",
    },
}

# Default mode
DEFAULT_MODE = "short"

# Server startup timeout in seconds
SERVER_STARTUP_TIMEOUT = 180.0

# Scripts to benchmark
BENCHMARK_SCRIPTS = [
    ("GigaAM (Transformers)", "run_gigaam.sh"),
    ("GigaAM (Native)", "run_gigaam_asr.sh"),
    ("Whisper Large V3 (Transformers)", "run_whisper_large_v3.sh"),
    ("WhisperX Large V3", "run_whisperx_large_v3.sh"),
]


@dataclass
class MemoryInfo:
    """Memory usage information."""
    process_memory_mb: float = 0.0  # Total process memory (RSS)
    model_memory_mb: float = 0.0    # Memory used by model
    gpu_memory_mb: float = 0.0      # GPU memory (if available)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    engine_name: str
    script_name: str
    transcription_time: float  # average time across iterations
    transcription_times: List[float] = field(default_factory=list)  # all iteration times
    text: str = ""
    text_length: int = 0
    audio_duration: float = 0.0
    speed_ratio: float = 0.0  # audio_duration / transcription_time
    iterations: int = 0
    success: bool = False
    error: Optional[str] = None
    memory: Optional[MemoryInfo] = None


def get_test_audio() -> Path:
    """Get the test audio file from sample-data directory."""
    audio_path = SAMPLE_DATA_DIR / TEST_AUDIO_FILE
    if not audio_path.exists():
        raise FileNotFoundError(f"Test audio file not found: {audio_path}")
    return audio_path


def get_audio_file_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(audio_path)
    return info.duration


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


def get_server_memory_info() -> Optional[MemoryInfo]:
    """
    Get memory information from server health endpoint.

    Returns:
        MemoryInfo or None if unavailable
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            memory_data = data.get("memory", {})
            return MemoryInfo(
                process_memory_mb=memory_data.get("process_memory_mb", 0.0),
                model_memory_mb=memory_data.get("model_memory_mb", 0.0),
                gpu_memory_mb=memory_data.get("gpu_memory_mb", 0.0),
            )
    except Exception as e:
        print(f"Failed to get memory info: {e}")
    return None


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


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def wait_for_port_free(port: int, timeout: float = 30.0) -> bool:
    """Wait until port is free."""
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def kill_process_on_port(port: int, exclude_pids: set = None) -> None:
    """Kill any process using the specified port (except excluded pids)."""
    if exclude_pids is None:
        exclude_pids = set()
    # Always exclude current process and parent
    exclude_pids.add(os.getpid())
    exclude_pids.add(os.getppid())

    try:
        # Use lsof to find process on port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    pid_int = int(pid)
                    # Don't kill excluded processes
                    if pid_int in exclude_pids:
                        continue
                    os.kill(pid_int, signal.SIGKILL)
                    print(f"Killed process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")


def stop_server(process: subprocess.Popen) -> None:
    """
    Stop the server process.

    Args:
        process: Server subprocess
    """
    if process is None:
        return

    server_pid = process.pid
    try:
        # Kill the entire process group
        try:
            pgid = os.getpgid(server_pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Wait for graceful shutdown
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill process group
            try:
                pgid = os.getpgid(server_pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
    except Exception as e:
        print(f"Error stopping server: {e}")

    # Wait a moment for port to be released naturally
    time.sleep(2)


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

    # Ensure port is free before starting
    if is_port_in_use(SERVER_PORT):
        print(f"Port {SERVER_PORT} is in use, waiting for it to be released...")
        if not wait_for_port_free(SERVER_PORT, timeout=15):
            print(f"Warning: Port {SERVER_PORT} still in use, trying to start anyway")

    # Start server process with new process group
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
    iterations: int = 5,
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

        # Verify we're talking to the right engine
        memory_info = None
        try:
            health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
            actual_engine = health.get("engine", "unknown")
            print(f"Server is ready! (engine: {actual_engine})")
        except Exception:
            print("Server is ready!")

        # Run transcription multiple times
        print(f"Transcribing {audio_duration:.1f}s audio ({iterations} iteration(s))...")
        transcription_times = []
        text = ""
        result = {}

        for i in range(iterations):
            start_time = time.time()
            try:
                result = transcribe_audio(audio_path)
            except Exception as e:
                print(f"Transcription request failed on iteration {i+1}: {e}")
                raise

            elapsed = time.time() - start_time
            transcription_times.append(elapsed)
            text = result.get("text", "")
            print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}s")

        # Get memory info after transcription (model is loaded)
        memory_info = get_server_memory_info()
        if memory_info:
            print(f"Memory usage:")
            print(f"  Process: {memory_info.process_memory_mb:.1f} MB")
            if memory_info.model_memory_mb > 0:
                print(f"  Model: {memory_info.model_memory_mb:.1f} MB")
            if memory_info.gpu_memory_mb > 0:
                print(f"  GPU: {memory_info.gpu_memory_mb:.1f} MB")

        # Calculate average
        avg_time = sum(transcription_times) / len(transcription_times)
        min_time = min(transcription_times)
        max_time = max(transcription_times)
        speed_ratio = audio_duration / avg_time if avg_time > 0 else 0

        print(f"Transcription completed:")
        print(f"  Average: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")
        print(f"  Speed: {speed_ratio:.2f}x realtime")
        print(f"  Text length: {len(text)} chars")
        if text:
            print(f"  Preview: {text[:100]}...")
        else:
            print("  Preview: (empty result)")
            print(f"  Full response: {result}")

        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=avg_time,
            transcription_times=transcription_times,
            text=text,
            text_length=len(text),
            audio_duration=audio_duration,
            speed_ratio=speed_ratio,
            iterations=iterations,
            success=True,
            memory=memory_info,
        )

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=0,
            transcription_times=[],
            text="",
            text_length=0,
            audio_duration=audio_duration,
            speed_ratio=0,
            iterations=0,
            success=False,
            error=f"Request error: {e}",
        )

    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            engine_name=engine_name,
            script_name=script_name,
            transcription_time=0,
            transcription_times=[],
            text="",
            text_length=0,
            audio_duration=audio_duration,
            speed_ratio=0,
            iterations=0,
            success=False,
            error=str(e),
        )

    finally:
        # Stop server
        if process is not None:
            print("Stopping server...")
            stop_server(process)

            # Wait for port to be released
            if not wait_for_port_free(SERVER_PORT, timeout=10):
                print(f"Warning: Port {SERVER_PORT} still in use, waiting more...")
                time.sleep(3)
                if is_port_in_use(SERVER_PORT):
                    print(f"Port still busy, will try to start anyway...")
            else:
                print("Port released.")


def print_results_table(results: list[BenchmarkResult], mode: str, audio_duration: float, iterations: int) -> None:
    """
    Print benchmark results as a formatted table.

    Args:
        results: List of benchmark results
        mode: Test mode name
        audio_duration: Duration of test audio
        iterations: Number of iterations per engine
    """
    print("\n")
    print("=" * 120)
    print(f"BENCHMARK RESULTS (mode: {mode}, {iterations} iteration(s), {audio_duration:.1f}s audio)")
    print("=" * 120)

    # Header
    print(f"{'Engine':<40} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Speed':<10} {'RAM':<12} {'GPU':<12} {'Status':<10}")
    print("-" * 120)

    # Results
    for r in results:
        if r.success:
            status = "✓ OK"
            avg_str = f"{r.transcription_time:.2f}"
            min_str = f"{min(r.transcription_times):.2f}" if r.transcription_times else "N/A"
            max_str = f"{max(r.transcription_times):.2f}" if r.transcription_times else "N/A"
            speed_str = f"{r.speed_ratio:.2f}x"
            # Memory info
            if r.memory and r.memory.model_memory_mb > 0:
                mem_str = f"{r.memory.model_memory_mb:.0f} MB"
            elif r.memory and r.memory.process_memory_mb > 0:
                mem_str = f"~{r.memory.process_memory_mb:.0f} MB"
            else:
                mem_str = "N/A"
            # GPU memory
            if r.memory and r.memory.gpu_memory_mb > 0:
                gpu_str = f"{r.memory.gpu_memory_mb:.0f} MB"
            else:
                gpu_str = "N/A"
        else:
            status = f"✗ {r.error[:15]}" if r.error else "✗ Failed"
            avg_str = "N/A"
            min_str = "N/A"
            max_str = "N/A"
            speed_str = "N/A"
            mem_str = "N/A"
            gpu_str = "N/A"

        print(f"{r.engine_name:<40} {avg_str:<10} {min_str:<10} {max_str:<10} {speed_str:<10} {mem_str:<12} {gpu_str:<12} {status:<10}")

    print("-" * 120)

    # Summary
    successful = [r for r in results if r.success]
    if successful:
        fastest = min(successful, key=lambda r: r.transcription_time)
        print(f"\nFastest: {fastest.engine_name}")
        print(f"  Average: {fastest.transcription_time:.2f}s")
        print(f"  Speed: {fastest.speed_ratio:.2f}x realtime")
        if fastest.memory and fastest.memory.model_memory_mb > 0:
            print(f"  RAM: {fastest.memory.model_memory_mb:.0f} MB")
        if fastest.memory and fastest.memory.gpu_memory_mb > 0:
            print(f"  GPU: {fastest.memory.gpu_memory_mb:.0f} MB")

    # Memory summary
    print("\nMemory Usage Summary:")
    for r in results:
        if r.success and r.memory:
            model_mem = r.memory.model_memory_mb if r.memory.model_memory_mb > 0 else r.memory.process_memory_mb
            gpu_mem = r.memory.gpu_memory_mb if r.memory.gpu_memory_mb > 0 else 0
            if gpu_mem > 0:
                print(f"  {r.engine_name}: {model_mem:.0f} MB RAM, {gpu_mem:.0f} MB GPU")
            else:
                print(f"  {r.engine_name}: {model_mem:.0f} MB RAM")

    print("\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OAITT ASR Engine Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  short   20s audio, 5 iterations - quick benchmark (default)
  long    60s audio, 3 iterations - tests chunked transcription
  full    full file (~137s), 1 iteration - complete longform test

Examples:
  python -m tests.test_benchmark                    # short mode
  python -m tests.test_benchmark --mode long        # long mode
  python -m tests.test_benchmark --mode full        # full file
  python -m tests.test_benchmark --iterations 10    # custom iterations
        """
    )
    parser.add_argument(
        "--mode", "-m",
        choices=list(TEST_MODES.keys()),
        default=DEFAULT_MODE,
        help=f"Test mode (default: {DEFAULT_MODE})"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=None,
        help="Override number of iterations"
    )
    parser.add_argument(
        "--script", "-s",
        action="append",
        default=None,
        help="Run only engines matching script name (e.g. run_gigaam_asr.sh). Repeatable."
    )
    return parser.parse_args()


def main():
    """Main benchmark runner."""
    args = parse_args()

    # Get mode configuration
    mode_config = TEST_MODES[args.mode]
    target_duration = mode_config["duration"]
    iterations = args.iterations if args.iterations is not None else mode_config["iterations"]

    print("=" * 80)
    print("OAITT ASR Engine Benchmark")
    print(f"Mode: {args.mode} - {mode_config['description']}")
    print("=" * 80)

    # Find and prepare audio
    print("\nPreparing test audio...")
    sample_path = get_test_audio()
    file_duration = get_audio_file_duration(sample_path)
    print(f"Using sample: {sample_path.name} (full duration: {file_duration:.1f}s)")

    # Determine actual duration to use
    if target_duration is None or target_duration >= file_duration:
        # Use full file
        audio_path = sample_path
        audio_duration = file_duration
        temp_file = False
        print(f"Using full file: {audio_duration:.1f}s")
    else:
        # Extract segment
        audio_path, audio_duration = extract_audio_segment(sample_path, target_duration)
        temp_file = True
        print(f"Extracted {audio_duration:.1f}s audio segment: {audio_path}")

    print(f"Iterations per engine: {iterations}")

    results = []

    scripts = BENCHMARK_SCRIPTS
    if args.script:
        wanted = set(args.script)
        scripts = [(n, s) for n, s in BENCHMARK_SCRIPTS if s in wanted]
        if not scripts:
            print(f"No matching scripts for: {args.script}")
            return

    try:
        # Run benchmarks for each engine
        for engine_name, script_name in scripts:
            # Check if script exists
            script_path = PROJECT_ROOT / script_name
            if not script_path.exists():
                print(f"\nSkipping {engine_name}: script {script_name} not found")
                results.append(BenchmarkResult(
                    engine_name=engine_name,
                    script_name=script_name,
                    transcription_time=0,
                    transcription_times=[],
                    text="",
                    text_length=0,
                    audio_duration=audio_duration,
                    speed_ratio=0,
                    iterations=0,
                    success=False,
                    error="Script not found",
                ))
                continue

            result = run_benchmark(engine_name, script_name, audio_path, audio_duration, iterations)
            results.append(result)

        # Print results table
        print_results_table(results, args.mode, audio_duration, iterations)

    finally:
        # Cleanup temp file only if we created one
        if temp_file and audio_path.exists() and audio_path != sample_path:
            audio_path.unlink()
            print(f"Cleaned up temporary file: {audio_path}")


if __name__ == "__main__":
    main()
