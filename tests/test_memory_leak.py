#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Тест для проверки утечек памяти в режиме GigaAM ASR.

Нарезает случайные кусочки аудио от 1 до 45 секунд из файла Sobolev_Andrey_1_0_00-2_17.ogg,
отправляет их на транскрипцию через API и мониторит потребление памяти.

Usage:
    python -m tests.test_memory_leak [--iterations N] [--batch-size B]

Examples:
    python -m tests.test_memory_leak                    # 1000 итераций
    python -m tests.test_memory_leak --iterations 500   # 500 итераций
    python -m tests.test_memory_leak --batch-size 10    # лог каждые 10 итераций
"""

import argparse
import os
import sys
import time
import signal
import socket
import subprocess
import tempfile
import random
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

import soundfile as sf
import requests
import psutil

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
MEMORY_STATS_ENDPOINT = f"{SERVER_URL}/admin/memory-stats"

# Default API token
API_TOKEN = "key"

# Server startup timeout in seconds
SERVER_STARTUP_TIMEOUT = 180.0

# Default script to run
DEFAULT_SCRIPT = "run_gigaam_asr.sh"


@dataclass
class MemorySample:
    """Sample of memory usage at a point in time."""
    iteration: int
    timestamp: float
    rss_mb: float
    vms_mb: float
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None


@dataclass
class TestResult:
    """Result of a single transcription."""
    iteration: int
    duration_sec: float
    transcription_time: float
    text_length: int
    success: bool
    error: Optional[str] = None


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


def extract_random_segment(input_path: Path, full_audio: tuple, min_sec: float = 1.0, max_sec: float = 45.0) -> tuple[Path, float]:
    """
    Extract a random segment from audio file.
    
    Args:
        input_path: Path to input audio file (for reference)
        full_audio: Tuple of (audio_data, sample_rate) - pre-loaded full audio
        min_sec: Minimum duration in seconds
        max_sec: Maximum duration in seconds
    
    Returns:
        Tuple of (Path to temporary file with extracted audio, actual duration)
    """
    audio_data, sample_rate = full_audio
    total_duration = len(audio_data) / sample_rate
    
    # Random duration between min and max
    duration = random.uniform(min_sec, min(max_sec, total_duration))
    
    # Random start position (ensure we don't go past the end)
    max_start = total_duration - duration
    start_sec = random.uniform(0, max(0, max_start))
    
    # Calculate sample positions
    start_sample = int(start_sec * sample_rate)
    end_sample = start_sample + int(duration * sample_rate)
    
    # Extract segment
    audio_segment = audio_data[start_sample:end_sample]
    actual_duration = len(audio_segment) / sample_rate
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_segment, sample_rate)
    
    return Path(temp_file.name), actual_duration


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
    exclude_pids.add(os.getpid())
    exclude_pids.add(os.getppid())

    try:
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
                    if pid_int in exclude_pids:
                        continue
                    os.kill(pid_int, signal.SIGKILL)
                    print(f"Killed process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")


def wait_for_server(timeout: float = SERVER_STARTUP_TIMEOUT, check_interval: float = 2.0) -> bool:
    """Wait for the server to become available."""
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


def start_server(script_name: str) -> Optional[subprocess.Popen]:
    """Start the server with the specified script."""
    script_path = PROJECT_ROOT / script_name
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return None
    
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
        preexec_fn=os.setsid,
    )
    
    return process


def stop_server(process: subprocess.Popen) -> None:
    """Stop the server process."""
    if process is None:
        return
    
    server_pid = process.pid
    try:
        try:
            pgid = os.getpgid(server_pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
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
    
    time.sleep(2)


def get_memory_stats() -> Optional[dict]:
    """Get memory stats from server."""
    try:
        response = requests.get(MEMORY_STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def transcribe_audio(audio_path: Path, language: str = "ru") -> tuple[bool, float, int, Optional[str]]:
    """
    Send audio file to transcription API.
    
    Returns:
        Tuple of (success, transcription_time, text_length, error)
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
    }
    
    start_time = time.time()
    try:
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
                timeout=600,
            )
        
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time
        text = result.get("text", "")
        return True, elapsed, len(text), None
        
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, 0, str(e)


def run_memory_leak_test(iterations: int = 1000, batch_size: int = 10, log_interval: int = 50) -> None:
    """
    Run memory leak test with random audio segments.
    
    Args:
        iterations: Number of transcriptions to perform
        batch_size: Number of transcriptions between memory snapshots
        log_interval: Number of iterations between progress logs
    """
    print("=" * 80)
    print("OAITT Memory Leak Test - GigaAM ASR")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Iterations: {iterations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Log interval: {log_interval}")
    print(f"  Script: {DEFAULT_SCRIPT}")
    print("=" * 80)
    
    # Load audio file
    print("\nLoading audio file...")
    sample_path = get_test_audio()
    full_duration = get_audio_file_duration(sample_path)
    print(f"Audio file: {sample_path.name}")
    print(f"Full duration: {full_duration:.1f}s")
    
    # Load full audio into memory for faster segment extraction
    print("Loading audio data into memory...")
    audio_data, sample_rate = sf.read(sample_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    full_audio = (audio_data, sample_rate)
    print(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
    
    process = None
    results: List[TestResult] = []
    memory_samples: List[MemorySample] = []
    temp_files: List[Path] = []
    
    try:
        # Start server
        print("\nStarting server...")
        process = start_server(DEFAULT_SCRIPT)
        
        if process is None:
            print("ERROR: Failed to start server")
            return
        
        print(f"Waiting for server (up to {SERVER_STARTUP_TIMEOUT}s)...")
        if not wait_for_server():
            print("ERROR: Server startup timeout")
            return
        
        try:
            health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
            actual_engine = health.get("engine", "unknown")
            print(f"Server is ready! (engine: {actual_engine})")
        except Exception:
            print("Server is ready!")
        
        # Get initial memory snapshot
        print("\nTaking initial memory snapshot...")
        initial_memory = get_memory_stats()
        if initial_memory:
            print(f"Initial memory: RSS={initial_memory.get('rss_mb', 0):.1f}MB, VMS={initial_memory.get('vms_mb', 0):.1f}MB")
        
        # Run transcriptions
        print(f"\nStarting {iterations} transcriptions...")
        print("-" * 80)
        
        start_time = time.time()
        
        for i in range(iterations):
            # Extract random segment
            audio_path, duration = extract_random_segment(sample_path, full_audio, min_sec=1.0, max_sec=45.0)
            temp_files.append(audio_path)
            
            # Transcribe
            success, trans_time, text_len, error = transcribe_audio(audio_path)
            
            result = TestResult(
                iteration=i + 1,
                duration_sec=duration,
                transcription_time=trans_time,
                text_length=text_len,
                success=success,
                error=error,
            )
            results.append(result)
            
            # Sample memory periodically
            if (i + 1) % batch_size == 0:
                mem_stats = get_memory_stats()
                if mem_stats:
                    memory_samples.append(MemorySample(
                        iteration=i + 1,
                        timestamp=time.time() - start_time,
                        rss_mb=mem_stats.get('rss_mb', 0),
                        vms_mb=mem_stats.get('vms_mb', 0),
                        tracemalloc_current_mb=mem_stats.get('tracemalloc_current_mb'),
                        tracemalloc_peak_mb=mem_stats.get('tracemalloc_peak_mb'),
                    ))
            
            # Log progress
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                success_count = sum(1 for r in results if r.success)
                avg_time = sum(r.transcription_time for r in results) / len(results) if results else 0
                
                mem_info = ""
                if memory_samples:
                    latest = memory_samples[-1]
                    mem_info = f" | Memory: RSS={latest.rss_mb:.1f}MB"
                
                print(f"[{i+1}/{iterations}] {elapsed:.1f}s elapsed | Success: {success_count}/{i+1} | Avg time: {avg_time:.2f}s{mem_info}")
                
                # Clean up old temp files to prevent disk space issues
                if len(temp_files) > 100:
                    for old_file in temp_files[:-50]:
                        try:
                            old_file.unlink()
                        except:
                            pass
                    temp_files = temp_files[-50:]
        
        total_elapsed = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        success_count = sum(1 for r in results if r.success)
        fail_count = len(results) - success_count
        
        print(f"\nTotal iterations: {iterations}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Average transcription time: {sum(r.transcription_time for r in results) / len(results):.2f}s")
        
        # Memory analysis
        if memory_samples:
            print("\n" + "-" * 80)
            print("MEMORY ANALYSIS")
            print("-" * 80)
            
            if initial_memory:
                initial_rss = initial_memory.get('rss_mb', 0)
            else:
                initial_rss = memory_samples[0].rss_mb if memory_samples else 0
            
            final_rss = memory_samples[-1].rss_mb if memory_samples else initial_rss
            
            print(f"Initial RSS: {initial_rss:.1f} MB")
            print(f"Final RSS: {final_rss:.1f} MB")
            print(f"RSS Growth: {final_rss - initial_rss:.1f} MB")
            print(f"RSS Growth per 100 iterations: {(final_rss - initial_rss) / iterations * 100:.1f} MB")
            
            # Memory trend
            if len(memory_samples) >= 2:
                first_half = memory_samples[:len(memory_samples)//2]
                second_half = memory_samples[len(memory_samples)//2:]
                first_avg = sum(s.rss_mb for s in first_half) / len(first_half)
                second_avg = sum(s.rss_mb for s in second_half) / len(second_half)
                
                print(f"\nFirst half avg RSS: {first_avg:.1f} MB")
                print(f"Second half avg RSS: {second_avg:.1f} MB")
                print(f"Difference: {second_avg - first_avg:.1f} MB")
                
                if second_avg > first_avg * 1.1:  # More than 10% growth
                    print("\n⚠️  WARNING: Possible memory leak detected!")
                    print(f"    Memory grew by {((second_avg / first_avg - 1) * 100):.1f}%")
                else:
                    print("\n✓ Memory usage appears stable")
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = PROJECT_ROOT / f"memory_leak_test_{timestamp}.json"
            
            test_data = {
                "timestamp": timestamp,
                "iterations": iterations,
                "batch_size": batch_size,
                "total_time_sec": total_elapsed,
                "success_count": success_count,
                "fail_count": fail_count,
                "initial_memory_mb": initial_rss,
                "final_memory_mb": final_rss,
                "memory_growth_mb": final_rss - initial_rss,
                "memory_samples": [
                    {
                        "iteration": s.iteration,
                        "timestamp": s.timestamp,
                        "rss_mb": s.rss_mb,
                        "vms_mb": s.vms_mb,
                        "tracemalloc_current_mb": s.tracemalloc_current_mb,
                        "tracemalloc_peak_mb": s.tracemalloc_peak_mb,
                    }
                    for s in memory_samples
                ],
                "failed_iterations": [
                    {"iteration": r.iteration, "error": r.error}
                    for r in results if not r.success
                ],
            }
            
            with open(results_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file}")
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        if process is not None:
            print("Stopping server...")
            stop_server(process)
            
            if not wait_for_port_free(SERVER_PORT, timeout=10):
                print(f"Warning: Port {SERVER_PORT} still in use")
            else:
                print("Port released.")
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
        print("Cleanup complete.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OAITT Memory Leak Test for GigaAM ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests.test_memory_leak                    # 1000 iterations (default)
  python -m tests.test_memory_leak --iterations 500   # 500 iterations
  python -m tests.test_memory_leak --batch-size 10    # Log every 10 iterations
  python -m tests.test_memory_leak --log-interval 25  # Progress every 25 iterations
        """
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=1000,
        help="Number of transcriptions to perform (default: 1000)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Number of transcriptions between memory snapshots (default: 10)"
    )
    parser.add_argument(
        "--log-interval", "-l",
        type=int,
        default=50,
        help="Number of iterations between progress logs (default: 50)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        run_memory_leak_test(
            iterations=args.iterations,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
        )
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
