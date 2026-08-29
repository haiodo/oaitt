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
import html
import json
import os
import platform
import sys
import time
import signal
import socket
import subprocess
import tempfile
from datetime import datetime
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

# Test modes configuration.
# min_seconds - минимальный бюджет замера на движок: итерации крутятся, пока
# суммарное время транскрипций не достигнет этого значения (при этом не меньше
# iterations прогонов). Быстрый движок так набирает больше выборок, медленный
# не гоняет лишнего - средние по одному-двум прогонам слишком шумные.
TEST_MODES = {
    "short": {
        "duration": 20.0,      # 20 seconds - within GigaAM short audio limit
        "iterations": 5,
        "min_seconds": 60.0,
        "description": "Quick benchmark with short audio",
    },
    "long": {
        "duration": 60.0,      # 60 seconds - tests chunked transcription
        "iterations": 3,
        "min_seconds": 60.0,
        "description": "Chunked transcription test",
    },
    "full": {
        "duration": None,      # None = use full file
        "iterations": 3,
        "min_seconds": 60.0,
        "description": "Full file longform test",
    },
}

# Default mode
DEFAULT_MODE = "short"

# Hard cap on iterations when filling the min_seconds budget. Быстрый движок
# иначе накрутил бы сотни прогонов ради 60 секунд бюджета; 60 замеров дают
# достаточно стабильное среднее (разброс на GigaAM - единицы процентов).
MAX_ITERATIONS = int(os.getenv("BENCHMARK_MAX_ITERATIONS", "60"))

# Server startup timeout in seconds. Щедрый, потому что первый запуск движка
# может тянуть веса с HuggingFace; мёртвый процесс всё равно ловится сразу.
SERVER_STARTUP_TIMEOUT = 600.0

# Single transcription request timeout. Зависший движок иначе вешает весь прогон,
# и отчёт не пишется вообще.
TRANSCRIBE_TIMEOUT = float(os.getenv("BENCHMARK_TRANSCRIBE_TIMEOUT", "300"))

# Scripts to benchmark.
# expected: ASR_ENGINE the server must report on /health - guards against
#           benchmarking a foreign server left on the port.
# punct:    whether the model emits punctuation (charwise CTC models do not).
BENCHMARK_SCRIPTS = [
    {"name": "GigaAM Native (CTC)", "script": "run_gigaam_asr.sh",
     "env": {"GIGAAM_MODEL": "v3_e2e_ctc"}, "expected": "gigaam", "punct": True},
    {"name": "GigaAM Native (RNNT)", "script": "run_gigaam_asr.sh",
     "env": {"GIGAAM_MODEL": "v3_e2e_rnnt"}, "expected": "gigaam", "punct": True},
    {"name": "GigaAM Multilingual (CTC)", "script": "run_gigaam_asr.sh",
     "env": {"GIGAAM_MODEL": "multilingual_ctc"}, "expected": "gigaam", "punct": False},
    {"name": "GigaAM Multilingual Large (CTC)", "script": "run_gigaam_asr.sh",
     "env": {"GIGAAM_MODEL": "multilingual_large_ctc"}, "expected": "gigaam", "punct": False},
    {"name": "GigaAM MLX (CTC)", "script": "run_gigaam_mlx_asr.sh",
     "env": {"GIGAAM_MLX_MODEL_TYPE": "ctc"}, "expected": "gigaam_mlx", "punct": True},
    {"name": "GigaAM MLX (RNNT)", "script": "run_gigaam_mlx_asr.sh",
     "env": {"GIGAAM_MLX_MODEL_TYPE": "rnnt"}, "expected": "gigaam_mlx", "punct": True},
    {"name": "GigaAM Swift (CTC)", "script": "run_swift_asr.sh",
     "env": {"GIGAAM_MLX_MODEL_TYPE": "ctc"}, "expected": "gigaam_mlx_swift", "punct": True},
    {"name": "GigaAM Swift (RNNT)", "script": "run_swift_asr.sh",
     "env": {"GIGAAM_MLX_MODEL_TYPE": "rnnt"}, "expected": "gigaam_mlx_swift", "punct": True},
    # MLX-порт есть только для large (600M) - малой 220M версии на HF нет.
    # int8/fp16 - квантизация одних и тех же весов, не разные размеры.
    {"name": "GigaAM Multilingual Large MLX (int8)", "script": "run_gigaam_multilingual_mlx.sh",
     "env": {"GIGAAM_ML_MLX_VARIANT": "int8"},
     "expected": "gigaam_multilingual_mlx", "punct": False},
    {"name": "GigaAM Multilingual Large MLX (fp16)", "script": "run_gigaam_multilingual_mlx.sh",
     "env": {"GIGAAM_ML_MLX_VARIANT": "fp16"},
     "expected": "gigaam_multilingual_mlx", "punct": False},
]

# Whisper-движки держим отдельно - они на порядок медленнее GigaAM на русском,
# а WhisperX к тому же зависает на длинном аудио. В набор по умолчанию не входят,
# запускать явно: ./benchmark.sh -s run_whisper_large_v3.sh
OPTIONAL_SCRIPTS = [
    # ai-sage/GigaAM-v3 custom modeling code is incompatible with transformers 5.x
    # (FeatureExtractor calls .item() on a meta tensor during instantiation).
    # Use the native gigaam engine instead - same model, much faster.
    {"name": "GigaAM (Transformers)", "script": "run_gigaam.sh",
     "env": {}, "expected": "transformers", "punct": True},
    {"name": "Whisper Large V3 (Transformers)", "script": "run_whisper_large_v3.sh",
     "env": {}, "expected": "transformers", "punct": True},
    {"name": "WhisperX Large V3", "script": "run_whisperx_large_v3.sh",
     "env": {}, "expected": "whisperx", "punct": True},
]


@dataclass
class MemoryInfo:
    """Memory usage information."""
    process_memory_mb: float = 0.0  # Total process memory (RSS)
    model_memory_mb: float = 0.0    # Memory used by model
    gpu_memory_mb: float = 0.0      # GPU memory (if available)
    gpu_peak_mb: float = 0.0        # Peak GPU memory across the run
    gpu_cache_mb: float = 0.0       # MLX allocator cache (reusable free buffers)
    peak_process_mb: float = 0.0    # Process RSS at the end of the run (steady state)


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
    punctuation: bool = True


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


_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Общая HTTP-сессия с keep-alive для всех запросов бенчмарка."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def reset_session() -> None:
    """Закрывает сессию - соединения к остановленному серверу больше не валидны."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None


def get_server_rss_mb() -> float:
    """
    RSS сервера на порту 9007 в МБ (сумма по процессу и его воркерам).

    /health отдаёт память только своего процесса и только по запросу - пик
    между итерациями так не поймать, поэтому опрашиваем ps напрямую.
    """
    try:
        pids = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True, text=True, timeout=5
        ).stdout.split()
        if not pids:
            return 0.0
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=5
        ).stdout.split()
        return sum(int(v) for v in out) / 1024  # ps reports KB
    except Exception:
        return 0.0


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
                gpu_peak_mb=memory_data.get("gpu_memory_peak_mb", 0.0),
                gpu_cache_mb=memory_data.get("gpu_memory_cache_mb", 0.0),
            )
    except Exception as e:
        print(f"Failed to get memory info: {e}")
    return None


def wait_for_server(
    timeout: float = SERVER_STARTUP_TIMEOUT,
    check_interval: float = 2.0,
    process: Optional[subprocess.Popen] = None,
) -> bool:
    """
    Wait for the server to become available.

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        process: Server process - if it exits early, stop waiting immediately

    Returns:
        True if server is available, False if timeout reached or process died
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        if process is not None and process.poll() is not None:
            return False
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


def start_server(script_name: str, env_overrides: Optional[dict] = None) -> Optional[subprocess.Popen]:
    """
    Start the server with the specified script.

    Args:
        script_name: Name of the script to run (e.g., "run_gigaam.sh")
        env_overrides: Extra environment variables for the server process

    Returns:
        Server subprocess or None if failed
    """
    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return None

    # Ensure port is free before starting. Оставшийся чужой сервер иначе ответит
    # на /health вместо нашего - и бенчмарк намерит не тот движок.
    if is_port_in_use(SERVER_PORT):
        print(f"Port {SERVER_PORT} is in use, waiting for it to be released...")
        if not wait_for_port_free(SERVER_PORT, timeout=15):
            print(f"Port {SERVER_PORT} still in use, killing the process holding it")
            kill_process_on_port(SERVER_PORT)
            if not wait_for_port_free(SERVER_PORT, timeout=10):
                print(f"Failed to free port {SERVER_PORT}")
                return None

    # Кеш результатов сделал бы замер бессмысленным: харнесс гоняет один и тот же файл,
    # и со второй итерации мерилось бы попадание в кеш, а не модель (145000x realtime).
    env = {**os.environ, "ASR_CACHE_SIZE": "0", **(env_overrides or {})}

    # Start server process with new process group
    process = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # Create new process group for clean termination
        env=env,
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

        # Переиспользуем одно соединение: сотни новых сокетов подряд копят
        # TIME_WAIT и упираются в лимит открытых файлов процесса.
        response = _session().post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=TRANSCRIBE_TIMEOUT,
        )

    response.raise_for_status()
    return response.json()


def run_benchmark(
    engine_name: str,
    script_name: str,
    audio_path: Path,
    audio_duration: float,
    iterations: int = 5,
    env_overrides: Optional[dict] = None,
    skip_warmup: bool = False,
    expected_engine: Optional[str] = None,
    min_seconds: float = 0.0,
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
    if env_overrides:
        print(f"Env: {env_overrides}")
    print(f"{'='*60}")

    process = None

    try:
        # Start server
        print("Starting server...")
        process = start_server(script_name, env_overrides)

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

        # Wait for server to be ready. Если скрипт запуска упал (нет зависимостей),
        # ждать полный таймаут смысла нет - процесс уже мёртв.
        print(f"Waiting for server (up to {SERVER_STARTUP_TIMEOUT}s)...")
        if not wait_for_server(process=process):
            err = "Server startup timeout"
            if process.poll() is not None:
                out = ""
                try:
                    out = process.stdout.read().decode("utf-8", "replace")[-500:]
                except Exception:
                    pass
                err = f"Server exited (code {process.returncode})"
                print(f"Server process exited with code {process.returncode}")
                if out:
                    print(f"Server output tail:\n{out}")
            return BenchmarkResult(
                engine_name=engine_name,
                script_name=script_name,
                transcription_time=0,
                text="",
                text_length=0,
                audio_duration=audio_duration,
                speed_ratio=0,
                success=False,
                error=err,
            )

        # Verify we're talking to the right engine. Чужой сервер на том же порту
        # (не добитый от прошлого движка) иначе намерит скорость не той модели.
        memory_info = None
        actual_engine = "unknown"
        try:
            health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
            actual_engine = health.get("engine", "unknown")
            print(f"Server is ready! (engine: {actual_engine})")
        except Exception:
            print("Server is ready!")

        if expected_engine and actual_engine != expected_engine:
            print(
                f"ERROR: expected engine '{expected_engine}' but /health reports "
                f"'{actual_engine}' - a foreign server is holding port {SERVER_PORT}"
            )
            return BenchmarkResult(
                engine_name=engine_name,
                script_name=script_name,
                transcription_time=0,
                text="",
                text_length=0,
                audio_duration=audio_duration,
                speed_ratio=0,
                success=False,
                error=f"Wrong engine: {actual_engine}",
            )

        # Первый прогон прогревает lazy-инициализацию движка (компиляция графа,
        # прогрев кэшей) и на порядок медленнее остальных - в замер не идёт.
        if not skip_warmup:
            print("Warmup run...")
            try:
                warm_start = time.time()
                transcribe_audio(audio_path)
                print(f"  Warmup: {time.time() - warm_start:.2f}s (excluded)")
            except requests.exceptions.Timeout:
                print(f"  Warmup timed out after {TRANSCRIBE_TIMEOUT:.0f}s - skipping engine")
                return BenchmarkResult(
                    engine_name=engine_name,
                    script_name=script_name,
                    transcription_time=0,
                    text="",
                    text_length=0,
                    audio_duration=audio_duration,
                    speed_ratio=0,
                    success=False,
                    error=f"Timeout after {TRANSCRIBE_TIMEOUT:.0f}s",
                )
            except Exception as e:
                print(f"  Warmup failed: {e}")

        # Run transcription until both the iteration count and the time budget
        # are satisfied - одиночные прогоны дают слишком шумное среднее.
        print(
            f"Transcribing {audio_duration:.1f}s audio "
            f"(min {iterations} iteration(s), min {min_seconds:.0f}s total)..."
        )
        transcription_times = []
        text = ""
        result = {}
        total_elapsed = 0.0
        i = 0

        while i < MAX_ITERATIONS:
            start_time = time.time()
            try:
                result = transcribe_audio(audio_path)
            except Exception as e:
                print(f"Transcription request failed on iteration {i+1}: {e}")
                raise

            elapsed = time.time() - start_time
            transcription_times.append(elapsed)
            total_elapsed += elapsed
            text = result.get("text", "")
            i += 1
            print(f"  Iteration {i}: {elapsed:.2f}s (total {total_elapsed:.1f}s)")

            if i >= iterations and total_elapsed >= min_seconds:
                break

        if i >= MAX_ITERATIONS and total_elapsed < min_seconds:
            print(
                f"  Reached the {MAX_ITERATIONS}-iteration cap at {total_elapsed:.1f}s "
                f"(budget was {min_seconds:.0f}s) - raise BENCHMARK_MAX_ITERATIONS to fill it"
            )

        # Память меряем один раз в конце: модель прогрета, аллокаторы вышли на
        # плато - это и есть установившееся потребление под нагрузкой.
        memory_info = get_server_memory_info()
        if memory_info:
            # RSS берём из /health: сервер знает свой PID точно, а внешний
            # lsof+ps ловит и клиентские сокеты, и гонку с остановкой процесса.
            memory_info.peak_process_mb = memory_info.process_memory_mb
            print(f"Memory usage:")
            print(f"  RSS: {memory_info.process_memory_mb:.1f} MB")
            if memory_info.model_memory_mb > 0:
                print(f"  Model: {memory_info.model_memory_mb:.1f} MB")
            if memory_info.gpu_memory_mb > 0:
                print(f"  GPU: {memory_info.gpu_memory_mb:.1f} MB")
            if memory_info.gpu_peak_mb > 0:
                print(f"  GPU peak: {memory_info.gpu_peak_mb:.1f} MB")

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
            iterations=len(transcription_times),
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
            reset_session()
            stop_server(process)

            # Wait for port to be released, then force-kill whatever still holds it -
            # иначе следующий движок отмерит скорость этого сервера.
            if not wait_for_port_free(SERVER_PORT, timeout=10):
                print(f"Port {SERVER_PORT} still in use, force-killing holder...")
                kill_process_on_port(SERVER_PORT)
                if not wait_for_port_free(SERVER_PORT, timeout=10):
                    print(f"Warning: port {SERVER_PORT} still busy")
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
    counts = sorted({r.iterations for r in results if r.success})
    iter_label = (
        f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    ) if counts else str(iterations)
    print(f"BENCHMARK RESULTS (mode: {mode}, {iter_label} iteration(s), {audio_duration:.1f}s audio)")
    print("=" * 120)

    # Header
    print(f"{'Engine':<36} {'Avg (s)':<9} {'Min (s)':<9} {'Max (s)':<9} {'Speed':<10} {'RSS':<11} {'GPU':<11} {'Punct':<7} {'Status':<10}")
    print("-" * 126)

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

        punct_str = "yes" if r.punctuation else "no"
        peak_str = f"{r.memory.peak_process_mb:.0f} MB" if r.memory and r.memory.peak_process_mb > 0 else "N/A"
        print(f"{r.engine_name:<36} {avg_str:<9} {min_str:<9} {max_str:<9} {speed_str:<10} {peak_str:<11} {gpu_str:<11} {punct_str:<7} {status:<10}")

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


def _host_info() -> dict:
    """Собирает описание машины для шапки отчёта."""
    def _sysctl(key: str) -> str:
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
        except Exception:
            return ""

    mem_bytes = _sysctl("hw.memsize")
    mem_gb = f"{int(mem_bytes) / (1024 ** 3):.0f} GB" if mem_bytes.isdigit() else ""
    return {
        "model": _sysctl("hw.model") or platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cores": _sysctl("hw.ncpu"),
        "memory": mem_gb,
        "platform": f"{platform.system()} {platform.release()}",
    }


def _result_rows(results: list[BenchmarkResult]) -> list[dict]:
    """Нормализует результаты в плоские строки для JSON/HTML."""
    rows = []
    for r in results:
        mem = r.memory
        rows.append({
            "engine": r.engine_name,
            "script": r.script_name,
            "avg_sec": round(r.transcription_time, 3) if r.success else None,
            "min_sec": round(min(r.transcription_times), 3) if r.transcription_times else None,
            "max_sec": round(max(r.transcription_times), 3) if r.transcription_times else None,
            "speed_ratio": round(r.speed_ratio, 2) if r.success else None,
            "ram_mb": round(mem.model_memory_mb or mem.process_memory_mb, 1) if mem else None,
            "peak_rss_mb": round(mem.peak_process_mb, 1) if mem and mem.peak_process_mb > 0 else None,
            "gpu_mb": round(mem.gpu_memory_mb, 1) if mem and mem.gpu_memory_mb > 0 else None,
            "gpu_peak_mb": round(mem.gpu_peak_mb, 1) if mem and mem.gpu_peak_mb > 0 else None,
            "gpu_cache_mb": round(mem.gpu_cache_mb, 1) if mem and mem.gpu_cache_mb > 0 else None,
            "text_length": r.text_length,
            "iterations": r.iterations,
            "punctuation": r.punctuation,
            "success": r.success,
            "error": r.error,
        })
    return rows


def write_json_report(path: Path, results, mode: str, audio_duration: float, iterations: int) -> None:
    """Сохраняет результаты бенчмарка в JSON."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "audio_duration_sec": round(audio_duration, 1),
        "iterations": iterations,
        "host": _host_info(),
        "results": _result_rows(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html_report(path: Path, results, mode: str, audio_duration: float, iterations: int) -> None:
    """Рендерит HTML-страницу с таблицей скорости, как в readme."""
    rows = _result_rows(results)
    host = _host_info()
    ok_rows = [r for r in rows if r["success"]]
    max_speed = max((r["speed_ratio"] for r in ok_rows), default=0) or 1
    fastest = min(ok_rows, key=lambda r: r["avg_sec"])["engine"] if ok_rows else None

    # Итераций у движков разное количество - добираются по времени, не по счётчику.
    counts = sorted({r["iterations"] for r in ok_rows}) or [iterations]
    iter_range = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"

    def fmt(value, suffix=""):
        return f"{value}{suffix}" if value is not None else "—"

    body = []
    for r in sorted(rows, key=lambda x: (not x["success"], -(x["speed_ratio"] or 0))):
        bar_pct = (r["speed_ratio"] or 0) / max_speed * 100
        cls = " class=\"best\"" if r["engine"] == fastest else ""
        status = "✓ OK" if r["success"] else f"✗ {html.escape(r['error'] or 'Failed')}"
        body.append(
            f"<tr{cls}>"
            f"<td>{html.escape(r['engine'])}</td>"
            f"<td>{fmt(r['avg_sec'])}</td><td>{fmt(r['min_sec'])}</td><td>{fmt(r['max_sec'])}</td>"
            f"<td class=\"speed\"><span class=\"bar\" style=\"width:{bar_pct:.1f}%\"></span>"
            f"<b>{fmt(r['speed_ratio'], 'x')}</b></td>"
            f"<td>{fmt(r['peak_rss_mb'] or r['ram_mb'], ' MB')}</td>"
            f"<td>{fmt(r['gpu_peak_mb'] or r['gpu_mb'], ' MB')}</td>"
            f"<td>{'✓' if r['punctuation'] else '—'}</td>"
            f"<td>{status}</td></tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OAITT ASR Benchmark</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; }}
 h1 {{ margin-bottom: .2rem; }}
 .meta {{ opacity: .7; font-size: .9em; margin-bottom: 1.5rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ padding: .5rem .6rem; border-bottom: 1px solid #8883; text-align: right; white-space: nowrap; }}
 th:first-child, td:first-child {{ text-align: left; white-space: normal; }}
 th {{ font-size: .85em; text-transform: uppercase; opacity: .65; }}
 tr.best td {{ font-weight: 600; }}
 .speed {{ position: relative; }}
 .bar {{ position: absolute; left: 0; top: 6px; bottom: 6px; background: #2ea04333; border-radius: 3px; z-index: -1; }}
 .wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>OAITT ASR Benchmark</h1>
<div class="meta">
 {html.escape(host['model'])} · {html.escape(host['chip'])} · {html.escape(host['cores'])} cores · {html.escape(host['memory'])}<br>
 Режим: <b>{html.escape(mode)}</b> · аудио {audio_duration:.1f}s · итераций: {iter_range} ·
 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
<div class="wrap"><table>
<thead><tr><th>Движок</th><th>Avg (s)</th><th>Min (s)</th><th>Max (s)</th>
<th>Скорость</th><th title="RSS процесса в конце прогона - установившееся потребление">RSS</th>
<th title="Пиковая GPU-память за прогон">GPU пик</th>
<th title="Модель проставляет пунктуацию">Пункт.</th><th>Статус</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody></table></div>
<p class="meta">Скорость = длительность аудио / среднее время транскрипции (x realtime).
RSS - память процесса в конце прогона по данным /health: модель прогрета,
аллокаторы вышли на плато (для планирования памяти ориентируйтесь на неё),
GPU пик - максимум одновременно живых буферов на GPU (unified memory на Apple Silicon).
Пункт. - модель проставляет пунктуацию (charwise CTC модели этого не делают).
<br>Замер идёт на одном файле. В проде MLX-движки держат больше: кэш аллокатора растёт
с числом уникальных длин чанков после VAD-split, а в lock-free режиме каждый параллельный
клиент держит свои активации. Кэш переиспользуется и в RSS не входит, но unified memory занимает.</p>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


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
        help="Run only engines matching script or engine name (e.g. run_gigaam_asr.sh). Repeatable."
    )
    parser.add_argument(
        "--min-seconds", type=float, default=None,
        help="Minimum total transcription time per engine (default: 60). "
             "Iterations keep running until this budget is filled."
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Do not run an untimed warmup transcription before measuring"
    )
    parser.add_argument("--json", default=None, help="Write results as JSON to this path")
    parser.add_argument("--html", default=None, help="Write results as HTML report to this path")
    return parser.parse_args()


def main():
    """Main benchmark runner."""
    args = parse_args()

    # Get mode configuration
    mode_config = TEST_MODES[args.mode]
    target_duration = mode_config["duration"]
    iterations = args.iterations if args.iterations is not None else mode_config["iterations"]
    min_seconds = (
        args.min_seconds if args.min_seconds is not None else mode_config["min_seconds"]
    )

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

    print(f"Iterations per engine: min {iterations}, min {min_seconds:.0f}s total")

    results = []

    scripts = BENCHMARK_SCRIPTS
    if args.script:
        wanted = set(args.script)
        scripts = [e for e in BENCHMARK_SCRIPTS + OPTIONAL_SCRIPTS
                   if e["script"] in wanted or e["name"] in wanted]
        if not scripts:
            print(f"No matching scripts for: {args.script}")
            return

    try:
        # Run benchmarks for each engine
        for entry in scripts:
            engine_name = entry["name"]
            script_name = entry["script"]
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
                    punctuation=entry["punct"],
                ))
                continue

            result = run_benchmark(
                engine_name, script_name, audio_path, audio_duration, iterations,
                entry["env"], args.no_warmup, entry["expected"], min_seconds
            )
            result.punctuation = entry["punct"]
            results.append(result)

        # Print results table
        print_results_table(results, args.mode, audio_duration, iterations)

        if args.json:
            write_json_report(Path(args.json), results, args.mode, audio_duration, iterations)
            print(f"JSON report: {args.json}")
        if args.html:
            write_html_report(Path(args.html), results, args.mode, audio_duration, iterations)
            print(f"HTML report: {args.html}")

    finally:
        # Cleanup temp file only if we created one
        if temp_file and audio_path.exists() and audio_path != sample_path:
            audio_path.unlink()
            print(f"Cleaned up temporary file: {audio_path}")


if __name__ == "__main__":
    main()
