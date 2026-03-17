#!/usr/bin/env python3
"""
OAITT Memory Leak Test - Simple version without tracemalloc overhead
"""

import os
import subprocess
import sys
import time
import signal
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SERVER_URL = "http://localhost:9007"
API_ENDPOINT = f"{SERVER_URL}/v1/audio/transcriptions"
API_TOKEN = "key"
TEST_AUDIO = PROJECT_ROOT / "sample-data" / "Sobolev_Andrey_1_0_00-2_17.ogg"


def kill_port_processes(port=9007):
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            for pid in result.stdout.strip().split('\n'):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except:
                    pass
    except:
        pass


def start_server():
    env = os.environ.copy()
    env.update({
        "ASR_ENGINE": "gigaam",
        "GIGAAM_MODEL": "v3_e2e_ctc",
        "PYTHONPATH": f"{PROJECT_ROOT}/vendor/gigaam:{env.get('PYTHONPATH', '')}",
    })
    
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def wait_for_server(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def get_memory_stats():
    try:
        response = requests.get(f"{SERVER_URL}/admin/memory-stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def send_request():
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": (TEST_AUDIO.name, f, "audio/wav")}
        data = {"model": "whisper-1", "language": "ru", "response_format": "verbose_json"}
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(API_ENDPOINT, headers=headers, files=files, data=data, timeout=600)
        return response.json()


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print(f"Memory leak test: {iterations} iterations")
    print("="*60)
    
    kill_port_processes(9007)
    time.sleep(2)
    
    process = start_server()
    
    try:
        print("Waiting for server...")
        if not wait_for_server():
            print("Server failed to start")
            return
        print("Server ready!")
        print()
        
        # Get baseline
        baseline = get_memory_stats()
        if baseline:
            print(f"Baseline RSS: {baseline.get('rss_mb', 0):.1f} MB")
        print()
        
        # Run iterations
        rss_values = []
        for i in range(1, iterations + 1):
            start = time.time()
            result = send_request()
            duration = time.time() - start
            
            # Get memory every 5 iterations using psutil directly
            if i % 5 == 0:
                import psutil
                current_process = psutil.Process()
                rss = current_process.memory_info().rss / 1024 / 1024
                rss_values.append((i, rss))
                print(f"Iteration {i:3d}: {duration:.1f}s, RSS: {rss:.1f} MB")
            else:
                print(f"Iteration {i:3d}: {duration:.1f}s", end='\r')
            
            time.sleep(0.5)
        
        print()
        print("="*60)
        
        # Analyze
        if len(rss_values) >= 2:
            first_i, first_rss = rss_values[0]
            last_i, last_rss = rss_values[-1]
            growth = last_rss - first_rss
            growth_per_iter = growth / (last_i - first_i)
            
            print(f"Memory growth: {growth:.1f} MB over {last_i - first_i} iterations")
            print(f"Growth per iteration: {growth_per_iter:.3f} MB")
            
            if growth_per_iter > 0.5:
                print("⚠️  WARNING: Significant leak detected!")
            elif growth_per_iter > 0.1:
                print("⚠️  Some leak detected")
            else:
                print("✓ Memory is stable")
        
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except:
            pass


if __name__ == "__main__":
    main()
