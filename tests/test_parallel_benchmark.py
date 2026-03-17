#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Бенчмарк параллельной пропускной способности.

Запускает сервер с заданным количеством MODEL_WORKERS и отправляет
параллельные запросы на транскрипцию, замеряя throughput и latency.

Сравнивает:
- 1 worker, 1 параллельный запрос (baseline)
- 1 worker, N параллельных запросов (contention)
- M workers, N параллельных запросов (parallel inference)

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python -m tests.test_parallel_benchmark [options]

    Options:
        --workers 1,2,4       Comma-separated worker counts to test (default: 1,2)
        --clients 1,2,4       Comma-separated client thread counts (default: 1,2,4)
        --duration 20         Audio duration in seconds (default: 20)
        --iterations 3        Iterations per client thread (default: 3)
        --samples-dir PATH    Directory with audio samples (default: samples/)
        --script SCRIPT       Server script to use (default: run_gigaam_asr.sh)

    Examples:
        python -m tests.test_parallel_benchmark
        python -m tests.test_parallel_benchmark --workers 1,2,4 --clients 1,2,4,8
        python -m tests.test_parallel_benchmark --duration 60 --iterations 5
        python -m tests.test_parallel_benchmark --samples-dir samples/ --script run_gigaam_asr.sh
"""

import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

SAMPLE_DATA_DIR = PROJECT_ROOT / "sample-data"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Fallback test audio files (tried in order)
TEST_AUDIO_CANDIDATES = [
    SAMPLE_DATA_DIR / "Sobolev_Andrey_1_0_00-2_17.ogg",
    *sorted(SAMPLES_DIR.glob("*.ogg")),
    *sorted(SAMPLES_DIR.glob("*.wav")),
]

SERVER_HOST = "localhost"
SERVER_PORT = 9007
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
API_ENDPOINT = f"{SERVER_URL}/v1/audio/transcriptions"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"

API_TOKEN = "key"
SERVER_STARTUP_TIMEOUT = 300.0

DEFAULT_SCRIPT = "run_gigaam_asr.sh"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    """Result of a single transcription request."""
    thread_id: int
    iteration: int
    elapsed: float
    text_length: int
    success: bool
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregated result for one (workers, clients) scenario."""
    num_workers: int
    num_clients: int
    iterations_per_client: int
    audio_duration_sec: float
    requests: List[RequestResult] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.requests if r.success]

    @property
    def total_requests(self) -> int:
        return len(self.requests)

    @property
    def success_count(self) -> int:
        return len(self.successful)

    @property
    def fail_count(self) -> int:
        return self.total_requests - self.success_count

    @property
    def avg_latency(self) -> float:
        s = self.successful
        return sum(r.elapsed for r in s) / len(s) if s else 0.0

    @property
    def min_latency(self) -> float:
        s = self.successful
        return min(r.elapsed for r in s) if s else 0.0

    @property
    def max_latency(self) -> float:
        s = self.successful
        return max(r.elapsed for r in s) if s else 0.0

    @property
    def p50_latency(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency(self) -> float:
        return self._percentile(95)

    @property
    def throughput_rps(self) -> float:
        """Requests per second based on wall time."""
        if self.wall_time <= 0:
            return 0.0
        return self.success_count / self.wall_time

    @property
    def throughput_audio_sec(self) -> float:
        """Audio seconds processed per wall-clock second."""
        if self.wall_time <= 0:
            return 0.0
        return (self.success_count * self.audio_duration_sec) / self.wall_time

    @property
    def speedup_vs_realtime(self) -> float:
        """How many times faster than realtime (total audio / wall time)."""
        return self.throughput_audio_sec

    def _percentile(self, pct: float) -> float:
        s = sorted(r.elapsed for r in self.successful)
        if not s:
            return 0.0
        idx = (pct / 100.0) * (len(s) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return s[lo]
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def find_test_audio(samples_dir: Optional[str] = None) -> Path:
    """Find a usable test audio file."""
    candidates = list(TEST_AUDIO_CANDIDATES)
    if samples_dir:
        d = Path(samples_dir)
        candidates = sorted(d.glob("*.ogg")) + sorted(d.glob("*.wav")) + candidates
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No test audio found. Place audio files in samples/ or sample-data/"
    )


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.duration


def extract_audio_segment(input_path: Path, duration_sec: float) -> tuple:
    """Extract a segment from audio. Returns (temp_path, actual_duration)."""
    audio_data, sample_rate = sf.read(str(input_path))
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    samples_needed = int(duration_sec * sample_rate)
    segment = audio_data[:samples_needed]
    actual = len(segment) / sample_rate
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, segment, sample_rate)
    return Path(tmp.name), actual


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def wait_for_port_free(port: int, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if not is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False


def kill_process_on_port(port: int) -> None:
    exclude = {os.getpid(), os.getppid()}
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            for pid_str in result.stdout.strip().split("\n"):
                try:
                    pid = int(pid_str)
                    if pid not in exclude:
                        os.kill(pid, signal.SIGKILL)
                        print(f"  Killed leftover process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass


def wait_for_server(timeout: float = SERVER_STARTUP_TIMEOUT) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(HEALTH_ENDPOINT, timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_server(script_name: str, num_workers: int) -> Optional[subprocess.Popen]:
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return None

    if is_port_in_use(SERVER_PORT):
        print(f"  Port {SERVER_PORT} in use, cleaning up...")
        kill_process_on_port(SERVER_PORT)
        if not wait_for_port_free(SERVER_PORT, timeout=15):
            print(f"  WARNING: port {SERVER_PORT} still in use")

    env = os.environ.copy()
    env["MODEL_WORKERS"] = str(num_workers)

    process = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    return process


def stop_server(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    pid = process.pid
    try:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                pgid = os.getpgid(pid)
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
        print(f"  Error stopping server: {e}")
    time.sleep(2)


# ---------------------------------------------------------------------------
# Transcription client
# ---------------------------------------------------------------------------


def transcribe_audio(audio_path: Path) -> dict:
    """Send one transcription request."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {
            "model": "whisper-1",
            "language": "ru",
            "response_format": "verbose_json",
        }
        resp = requests.post(
            API_ENDPOINT, headers=headers, files=files, data=data, timeout=600,
        )
    resp.raise_for_status()
    return resp.json()


def run_client_thread(
    thread_id: int,
    audio_path: Path,
    iterations: int,
    results: list,
    lock: threading.Lock,
    barrier: threading.Barrier,
):
    """Worker function for one client thread. Waits on barrier then fires requests."""
    # Wait for all threads to be ready
    barrier.wait()

    for i in range(iterations):
        start = time.perf_counter()
        try:
            resp = transcribe_audio(audio_path)
            elapsed = time.perf_counter() - start
            text = resp.get("text", "")
            result = RequestResult(
                thread_id=thread_id,
                iteration=i,
                elapsed=elapsed,
                text_length=len(text),
                success=True,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            result = RequestResult(
                thread_id=thread_id,
                iteration=i,
                elapsed=elapsed,
                text_length=0,
                success=False,
                error=str(e),
            )
        with lock:
            results.append(result)
            status = "OK" if result.success else f"FAIL: {result.error}"
            print(
                f"    [thread {thread_id}] iter {i+1}/{iterations}: "
                f"{result.elapsed:.2f}s — {status}"
            )


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def warmup_server(audio_path: Path, count: int = 1) -> None:
    """Send warmup requests so JIT/caches are primed."""
    print(f"  Warming up ({count} request(s))...")
    for i in range(count):
        try:
            transcribe_audio(audio_path)
        except Exception as e:
            print(f"  Warmup request {i+1} failed: {e}")


def run_scenario(
    audio_path: Path,
    audio_duration_sec: float,
    num_clients: int,
    iterations_per_client: int,
    num_workers: int,
) -> ScenarioResult:
    """Run one scenario: fire `num_clients` threads each doing `iterations_per_client`."""
    scenario = ScenarioResult(
        num_workers=num_workers,
        num_clients=num_clients,
        iterations_per_client=iterations_per_client,
        audio_duration_sec=audio_duration_sec,
    )

    results: list[RequestResult] = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_clients)

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=num_clients) as pool:
        futures = []
        for tid in range(num_clients):
            f = pool.submit(
                run_client_thread,
                thread_id=tid,
                audio_path=audio_path,
                iterations=iterations_per_client,
                results=results,
                lock=lock,
                barrier=barrier,
            )
            futures.append(f)
        # Wait for all threads to finish
        for f in as_completed(futures):
            f.result()  # propagate exceptions

    wall_end = time.perf_counter()

    scenario.requests = results
    scenario.wall_time = wall_end - wall_start
    return scenario


def run_full_benchmark(
    script_name: str,
    audio_path: Path,
    audio_duration_sec: float,
    worker_counts: List[int],
    client_counts: List[int],
    iterations_per_client: int,
) -> List[ScenarioResult]:
    """Run all (workers × clients) scenarios."""
    all_results: List[ScenarioResult] = []

    for num_workers in worker_counts:
        print(f"\n{'='*70}")
        print(f"Starting server with MODEL_WORKERS={num_workers}")
        print(f"{'='*70}")

        process = start_server(script_name, num_workers)
        if process is None:
            print(f"  SKIP: could not start server")
            for nc in client_counts:
                all_results.append(ScenarioResult(
                    num_workers=num_workers,
                    num_clients=nc,
                    iterations_per_client=iterations_per_client,
                    audio_duration_sec=audio_duration_sec,
                ))
            continue

        try:
            print(f"  Waiting for server (timeout={SERVER_STARTUP_TIMEOUT}s)...")
            if not wait_for_server():
                print(f"  SKIP: server startup timeout")
                for nc in client_counts:
                    all_results.append(ScenarioResult(
                        num_workers=num_workers,
                        num_clients=nc,
                        iterations_per_client=iterations_per_client,
                        audio_duration_sec=audio_duration_sec,
                    ))
                continue

            # Print server info
            try:
                health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
                engine = health.get("engine", "unknown")
                print(f"  Server ready (engine={engine})")
                pool_info = health.get("model", {})
                if isinstance(pool_info, dict) and "pool_size" in pool_info:
                    print(f"  Pool size: {pool_info['pool_size']}")
            except Exception:
                print(f"  Server ready")

            # Warmup
            warmup_server(audio_path, count=1)

            # Run scenarios for each client count
            for num_clients in client_counts:
                print(f"\n  --- {num_workers} worker(s), {num_clients} client(s), "
                      f"{iterations_per_client} iter(s) each ---")

                scenario = run_scenario(
                    audio_path=audio_path,
                    audio_duration_sec=audio_duration_sec,
                    num_clients=num_clients,
                    iterations_per_client=iterations_per_client,
                    num_workers=num_workers,
                )
                all_results.append(scenario)

                if scenario.success_count > 0:
                    print(f"  Results: {scenario.success_count}/{scenario.total_requests} OK, "
                          f"wall={scenario.wall_time:.2f}s, "
                          f"avg_lat={scenario.avg_latency:.2f}s, "
                          f"throughput={scenario.throughput_rps:.2f} req/s, "
                          f"{scenario.throughput_audio_sec:.1f}x realtime")
                else:
                    print(f"  Results: ALL FAILED ({scenario.fail_count} requests)")

        finally:
            print(f"\n  Stopping server (workers={num_workers})...")
            stop_server(process)
            if not wait_for_port_free(SERVER_PORT, timeout=15):
                print(f"  Warning: port still in use, forcing cleanup")
                kill_process_on_port(SERVER_PORT)
                time.sleep(3)

    return all_results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results_table(results: List[ScenarioResult]) -> None:
    """Print a formatted comparison table."""
    print("\n")
    print("=" * 130)
    print("PARALLEL BENCHMARK RESULTS")
    print("=" * 130)

    # Header
    cols = [
        ("Workers", 8),
        ("Clients", 8),
        ("OK/Total", 10),
        ("Wall (s)", 10),
        ("Avg Lat", 10),
        ("P50 Lat", 10),
        ("P95 Lat", 10),
        ("Min Lat", 10),
        ("Max Lat", 10),
        ("Req/s", 8),
        ("Audio×RT", 10),
    ]
    header = "".join(f"{name:<{width}}" for name, width in cols)
    print(header)
    print("-" * 130)

    # Find baseline (1 worker, 1 client)
    baseline_throughput = None
    for r in results:
        if r.num_workers == 1 and r.num_clients == 1 and r.success_count > 0:
            baseline_throughput = r.throughput_rps
            break

    for r in results:
        if r.success_count == 0:
            vals = [
                f"{r.num_workers}",
                f"{r.num_clients}",
                f"0/{r.total_requests}",
                "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
            ]
        else:
            speedup_str = f"{r.throughput_audio_sec:.1f}x"
            if baseline_throughput and baseline_throughput > 0:
                rel = r.throughput_rps / baseline_throughput
                speedup_str += f" ({rel:.1f}×)"

            vals = [
                f"{r.num_workers}",
                f"{r.num_clients}",
                f"{r.success_count}/{r.total_requests}",
                f"{r.wall_time:.2f}",
                f"{r.avg_latency:.2f}",
                f"{r.p50_latency:.2f}",
                f"{r.p95_latency:.2f}",
                f"{r.min_latency:.2f}",
                f"{r.max_latency:.2f}",
                f"{r.throughput_rps:.2f}",
                speedup_str,
            ]

        row = "".join(f"{v:<{w}}" for v, (_, w) in zip(vals, cols))
        print(row)

    print("-" * 130)

    # Summary
    successful = [r for r in results if r.success_count > 0]
    if successful:
        best = max(successful, key=lambda r: r.throughput_rps)
        print(
            f"\nBest throughput: {best.num_workers} workers, "
            f"{best.num_clients} clients → "
            f"{best.throughput_rps:.2f} req/s, "
            f"{best.throughput_audio_sec:.1f}x realtime"
        )
        if baseline_throughput and baseline_throughput > 0:
            print(
                f"Improvement over baseline (1w/1c): "
                f"{best.throughput_rps / baseline_throughput:.2f}×"
            )

    print()


def print_latency_breakdown(results: List[ScenarioResult]) -> None:
    """Print per-thread latency breakdown for each scenario."""
    for r in results:
        if not r.successful:
            continue
        print(f"\n--- Workers={r.num_workers}, Clients={r.num_clients} ---")
        # Group by thread
        by_thread: dict[int, list[RequestResult]] = {}
        for req in r.requests:
            by_thread.setdefault(req.thread_id, []).append(req)

        for tid in sorted(by_thread.keys()):
            reqs = sorted(by_thread[tid], key=lambda x: x.iteration)
            times = [f"{req.elapsed:.2f}s" for req in reqs if req.success]
            avg = (
                sum(req.elapsed for req in reqs if req.success)
                / max(1, sum(1 for req in reqs if req.success))
            )
            print(f"  Thread {tid}: {', '.join(times)}  (avg {avg:.2f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="OAITT Parallel Throughput Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tests.test_parallel_benchmark
  python -m tests.test_parallel_benchmark --workers 1,2,4 --clients 1,2,4
  python -m tests.test_parallel_benchmark --workers 1,4 --clients 4 --iterations 5
  python -m tests.test_parallel_benchmark --duration 60 --script run_gigaam_asr.sh
        """,
    )
    parser.add_argument(
        "--workers", "-w", type=str, default="1,2",
        help="Comma-separated MODEL_WORKERS counts to test (default: 1,2)",
    )
    parser.add_argument(
        "--clients", "-c", type=str, default="1,2,4",
        help="Comma-separated number of parallel client threads (default: 1,2,4)",
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=20.0,
        help="Audio duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=3,
        help="Iterations per client thread (default: 3)",
    )
    parser.add_argument(
        "--samples-dir", type=str, default=None,
        help="Directory with audio samples (default: auto-detect)",
    )
    parser.add_argument(
        "--script", "-s", type=str, default=DEFAULT_SCRIPT,
        help=f"Server launch script (default: {DEFAULT_SCRIPT})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-thread latency breakdown",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    worker_counts = parse_int_list(args.workers)
    client_counts = parse_int_list(args.clients)

    print("=" * 70)
    print("OAITT Parallel Throughput Benchmark")
    print("=" * 70)
    print(f"  Script:     {args.script}")
    print(f"  Workers:    {worker_counts}")
    print(f"  Clients:    {client_counts}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Iterations: {args.iterations} per client")
    print()

    # Find audio
    source = find_test_audio(args.samples_dir)
    source_duration = get_audio_duration(source)
    print(f"  Audio source: {source.name} ({source_duration:.1f}s)")

    # Extract segment if needed
    temp_file = None
    if args.duration and args.duration < source_duration:
        audio_path, actual_duration = extract_audio_segment(source, args.duration)
        temp_file = audio_path
        print(f"  Extracted {actual_duration:.1f}s segment → {audio_path}")
    else:
        audio_path = source
        actual_duration = source_duration
        print(f"  Using full file: {actual_duration:.1f}s")

    total_scenarios = len(worker_counts) * len(client_counts)
    total_requests = sum(c * args.iterations for c in client_counts) * len(worker_counts)
    print(f"\n  Scenarios: {total_scenarios}")
    print(f"  Total requests: {total_requests}")
    print(f"  Estimated audio processed: {total_requests * actual_duration:.0f}s")

    try:
        results = run_full_benchmark(
            script_name=args.script,
            audio_path=audio_path,
            audio_duration_sec=actual_duration,
            worker_counts=worker_counts,
            client_counts=client_counts,
            iterations_per_client=args.iterations,
        )

        print_results_table(results)

        if args.verbose:
            print_latency_breakdown(results)

    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()
            print(f"Cleaned up temp file: {temp_file}")


if __name__ == "__main__":
    main()
