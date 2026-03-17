#!/usr/bin/env python3
"""
OAITT Memory Leak Analyzer

Скрипт для детального анализа утечек памяти.
Сохраняет снапшоты памяти между итерациями и сравнивает их.

Usage:
    python scripts/analyze_memory_leaks.py --iterations 50 --snapshot-every 5

"""

import argparse
import os
import subprocess
import sys
import time
import signal
import tracemalloc
from pathlib import Path
from datetime import datetime
import requests

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
SERVER_URL = "http://localhost:9007"
API_ENDPOINT = f"{SERVER_URL}/v1/audio/transcriptions"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
API_TOKEN = "key"
TEST_AUDIO = PROJECT_ROOT / "sample-data" / "Sobolev_Andrey_1_0_00-2_17.ogg"


def kill_port_processes(port: int = 9007):
    """Kill processes using the specified port."""
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
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Killed process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"Warning: Could not check/kill port {port}: {e}")


def wait_for_server(timeout: float = 60.0) -> bool:
    """Wait for server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def start_server() -> subprocess.Popen:
    """Start the GigaAM server with memory monitoring."""
    print("Starting GigaAM server...")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        "ASR_ENGINE": "gigaam",
        "GIGAAM_MODEL": "v3_e2e_ctc",
        "MEMORY_LOG_ENABLED": "true",
        "MEMORY_LOG_INTERVAL": "5",
        "MEMORY_LOG_TOP_ALLOCATIONS": "10",
        "PYTHONPATH": f"{PROJECT_ROOT}/vendor/gigaam:{env.get('PYTHONPATH', '')}",
        "TIMEOUT_ENABLED": "true",
    })
    
    # Start server
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    
    return process


def stop_server(process: subprocess.Popen):
    """Stop the server gracefully."""
    if process is None:
        return
    
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except:
        pass
    
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except:
            pass


def send_transcription_request() -> dict:
    """Send a transcription request to the server."""
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": (TEST_AUDIO.name, f, "audio/wav")}
        data = {
            "model": "whisper-1",
            "language": "ru",
            "response_format": "verbose_json",
        }
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()


def analyze_memory_growth(snapshots: list, output_dir: Path):
    """Analyze memory growth between snapshots."""
    print("\n" + "="*60)
    print("MEMORY GROWTH ANALYSIS")
    print("="*60)
    
    if len(snapshots) < 2:
        print("Not enough snapshots for analysis")
        return
    
    # Compare first and last snapshot
    first_snapshot = snapshots[0]
    last_snapshot = snapshots[-1]
    
    top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')
    
    print(f"\nTotal snapshots: {len(snapshots)}")
    print(f"\nTop memory growth between first and last snapshot:")
    print("-" * 60)
    
    for stat in top_stats[:15]:
        print(f"{stat}")
    
    # Save detailed comparison
    comparison_file = output_dir / "memory_comparison.txt"
    with open(comparison_file, "w") as f:
        f.write("Memory Growth Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total snapshots: {len(snapshots)}\n\n")
        f.write("Top memory growth:\n")
        f.write("-" * 60 + "\n")
        for stat in top_stats[:50]:
            f.write(f"{stat}\n")
    
    print(f"\nDetailed comparison saved to: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="Memory leak analyzer for OAITT")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--snapshot-every", type=int, default=5, help="Take snapshot every N iterations")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for results")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "logs" / f"memory_analysis_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("OAITT Memory Leak Analyzer")
    print("="*60)
    print(f"Iterations: {args.iterations}")
    print(f"Snapshot every: {args.snapshot_every} iterations")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check test audio exists
    if not TEST_AUDIO.exists():
        print(f"Error: Test audio not found: {TEST_AUDIO}")
        sys.exit(1)
    
    # Kill any existing processes on port 9007
    kill_port_processes(9007)
    time.sleep(2)
    
    # Start tracemalloc in main process
    tracemalloc.start()
    print("✓ Started tracemalloc in analyzer process")
    
    # Start server
    server_process = start_server()
    
    try:
        # Wait for server
        print("Waiting for server to be ready...")
        if not wait_for_server():
            print("✗ Server failed to start!")
            # Read server output
            stdout, _ = server_process.communicate(timeout=5)
            print("Server output:")
            print(stdout.decode() if stdout else "(no output)")
            sys.exit(1)
        print("✓ Server is ready!")
        
        # Take initial snapshot
        snapshots = []
        print("\nTaking initial memory snapshot...")
        snapshots.append(tracemalloc.take_snapshot())
        
        # Run iterations
        print(f"\nRunning {args.iterations} iterations...")
        print("-" * 60)
        
        for i in range(1, args.iterations + 1):
            start_time = time.time()
            
            try:
                result = send_transcription_request()
                duration = time.time() - start_time
                print(f"Iteration {i}/{args.iterations}: {duration:.1f}s - {len(result.get('text', ''))} chars")
            except Exception as e:
                print(f"Iteration {i}/{args.iterations}: ERROR - {e}")
                continue
            
            # Take snapshot if needed
            if i % args.snapshot_every == 0:
                # Client snapshot
                snapshot = tracemalloc.take_snapshot()
                snapshots.append(snapshot)
                snapshot_file = args.output_dir / f"snapshot_{i:04d}.snapshot"
                snapshot.dump(str(snapshot_file))
                print(f"  ↳ Client snapshot saved")
                
                # Server snapshot
                try:
                    server_snapshot_path = f"{args.output_dir}/server_snapshot_{i:04d}.snapshot"
                    response = requests.post(
                        f"{SERVER_URL}/admin/memory-snapshot",
                        params={"filepath": str(server_snapshot_path)},
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            print(f"  ↳ Server snapshot saved ({result.get('size_bytes', 0) // 1024} KB)")
                        else:
                            print(f"  ↳ Server snapshot failed: {result.get('message')}")
                    else:
                        print(f"  ↳ Server snapshot failed: HTTP {response.status_code}")
                except Exception as e:
                    print(f"  ↳ Server snapshot error: {e}")
            
            # Small delay between requests
            time.sleep(1)
        
        print("-" * 60)
        print("\n✓ All iterations completed!")
        
        # Analyze memory growth
        analyze_memory_growth(snapshots, args.output_dir)
        
        # Take final snapshot
        final_snapshot = tracemalloc.take_snapshot()
        final_file = args.output_dir / "snapshot_final.snapshot"
        final_snapshot.dump(str(final_file))
        print(f"\nFinal snapshot saved to: {final_file}")
        
        # Calculate memory statistics
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nMemory statistics:")
        print(f"  Current memory: {current / 1024 / 1024:.1f} MB")
        print(f"  Peak memory: {peak / 1024 / 1024:.1f} MB")
        
    finally:
        # Stop server
        print("\nStopping server...")
        stop_server(server_process)
        
        # Stop tracemalloc
        tracemalloc.stop()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
