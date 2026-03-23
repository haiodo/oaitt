#!/usr/bin/env python3
"""
Детальный тест утечки памяти с профилированием конкретных участков кода.
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
import tracemalloc
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

import soundfile as sf
import requests
import psutil
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_DATA_DIR = PROJECT_ROOT / "sample-data"
TEST_AUDIO_FILE = "Sobolev_Andrey_1_0_00-2_17.ogg"

SERVER_HOST = "localhost"
SERVER_PORT = 9007
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
API_ENDPOINT = f"{SERVER_URL}/v1/audio/transcriptions"
HEALTH_ENDPOINT = f"{SERVER_URL}/health"
MEMORY_STATS_ENDPOINT = f"{SERVER_URL}/admin/memory-stats"

API_TOKEN = "key"
SERVER_STARTUP_TIMEOUT = 180.0


@dataclass
class DetailedMemorySample:
    """Детальный сэмпл памяти."""
    iteration: int
    timestamp: float
    rss_mb: float
    vms_mb: float
    tracemalloc_current_mb: float
    tracemalloc_peak_mb: float
    # Топ allocation sites только из нашего кода
    top_allocations: List[Dict] = field(default_factory=list)


def get_memory_stats() -> Optional[dict]:
    """Get memory stats from server."""
    try:
        response = requests.get(MEMORY_STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def extract_random_segment(input_path: Path, full_audio: tuple, min_sec: float = 1.0, max_sec: float = 45.0) -> tuple[Path, float]:
    """Extract a random segment from audio file."""
    audio_data, sample_rate = full_audio
    total_duration = len(audio_data) / sample_rate
    
    duration = random.uniform(min_sec, min(max_sec, total_duration))
    max_start = total_duration - duration
    start_sec = random.uniform(0, max(0, max_start))
    
    start_sample = int(start_sec * sample_rate)
    end_sample = start_sample + int(duration * sample_rate)
    
    audio_segment = audio_data[start_sample:end_sample]
    actual_duration = len(audio_segment) / sample_rate
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_segment, sample_rate)
    
    return Path(temp_file.name), actual_duration


def transcribe_audio(audio_path: Path, language: str = "ru") -> tuple[bool, float, int, Optional[str]]:
    """Send audio file to transcription API."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
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


def analyze_trace(snapshot: tracemalloc.Snapshot, top_n: int = 10) -> List[Dict]:
    """Analyze tracemalloc snapshot, filter for our code only."""
    top_stats = snapshot.statistics('lineno')
    
    our_allocations = []
    for stat in top_stats[:top_n]:
        filename = stat.traceback.format()[-1] if stat.traceback.format() else "unknown"
        # Фильтруем только наш код
        if any(x in filename for x in ['/src/', '/vendor/gigaam/', 'test_memory']):
            our_allocations.append({
                'file': filename,
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count,
            })
    
    return our_allocations


def run_detailed_test(iterations: int = 100, delay_between: float = 0.0) -> None:
    """
    Run detailed memory test with tracemalloc profiling.
    
    Args:
        iterations: Number of transcriptions
        delay_between: Delay between transcriptions in seconds (to test if MPS needs time)
    """
    print("=" * 80)
    print("DETAILED MEMORY LEAK ANALYSIS")
    print("=" * 80)
    print(f"Iterations: {iterations}")
    print(f"Delay between requests: {delay_between}s")
    
    # Load audio
    print("\nLoading audio...")
    sample_path = SAMPLE_DATA_DIR / TEST_AUDIO_FILE
    audio_data, sample_rate = sf.read(sample_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    full_audio = (audio_data, sample_rate)
    print(f"Audio loaded: {len(audio_data)} samples")
    
    # Start tracemalloc
    print("\nStarting tracemalloc...")
    tracemalloc.start()
    
    samples: List[DetailedMemorySample] = []
    temp_files: List[Path] = []
    
    try:
        # Initial snapshot
        print("Taking initial snapshot...")
        initial_snapshot = tracemalloc.take_snapshot()
        initial_mem = get_memory_stats()
        
        if initial_mem:
            print(f"Initial RSS: {initial_mem.get('rss_mb', 0):.1f} MB")
            print(f"Initial VMS: {initial_mem.get('vms_mb', 0):.1f} MB")
        
        print(f"\nRunning {iterations} transcriptions...")
        print("-" * 80)
        
        start_time = time.time()
        
        for i in range(iterations):
            # Extract and transcribe
            audio_path, duration = extract_random_segment(sample_path, full_audio)
            temp_files.append(audio_path)
            
            success, trans_time, text_len, error = transcribe_audio(audio_path)
            
            if not success:
                print(f"  ERROR at iteration {i+1}: {error}")
            
            # Delay if specified (for MPS cleanup)
            if delay_between > 0:
                time.sleep(delay_between)
            
            # Sample memory every 5 iterations
            if (i + 1) % 5 == 0:
                mem_stats = get_memory_stats()
                current, peak = tracemalloc.get_traced_memory()
                snapshot = tracemalloc.take_snapshot()
                
                sample = DetailedMemorySample(
                    iteration=i + 1,
                    timestamp=time.time() - start_time,
                    rss_mb=mem_stats.get('rss_mb', 0) if mem_stats else 0,
                    vms_mb=mem_stats.get('vms_mb', 0) if mem_stats else 0,
                    tracemalloc_current_mb=current / (1024 * 1024),
                    tracemalloc_peak_mb=peak / (1024 * 1024),
                    top_allocations=analyze_trace(snapshot),
                )
                samples.append(sample)
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"[{i+1}/{iterations}] {elapsed:.1f}s | "
                      f"RSS: {sample.rss_mb:.0f}MB | "
                      f"Python: {sample.tracemalloc_current_mb:.1f}MB")
            
            # Cleanup old temp files
            if len(temp_files) > 50:
                for old_file in temp_files[:-25]:
                    try:
                        old_file.unlink()
                    except:
                        pass
                temp_files = temp_files[-25:]
        
        total_elapsed = time.time() - start_time
        
        # Analysis
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS")
        print("=" * 80)
        
        if samples:
            initial_rss = samples[0].rss_mb
            final_rss = samples[-1].rss_mb
            initial_python = samples[0].tracemalloc_current_mb
            final_python = samples[-1].tracemalloc_current_mb
            
            print(f"\nProcess Memory (RSS):")
            print(f"  Start: {initial_rss:.1f} MB")
            print(f"  End:   {final_rss:.1f} MB")
            print(f"  Delta: {final_rss - initial_rss:+.1f} MB")
            print(f"  Per iteration: {(final_rss - initial_rss) / iterations:.2f} MB")
            
            print(f"\nPython Objects (tracemalloc):")
            print(f"  Start: {initial_python:.1f} MB")
            print(f"  End:   {final_python:.1f} MB")
            print(f"  Delta: {final_python - initial_python:+.1f} MB")
            
            # Key insight
            rss_growth = final_rss - initial_rss
            python_growth = final_python - initial_python
            native_growth = rss_growth - python_growth
            
            print(f"\n=== KEY FINDINGS ===")
            print(f"RSS Growth:       {rss_growth:+.1f} MB")
            print(f"Python Growth:    {python_growth:+.1f} MB")
            print(f"Native/MPS Growth: {native_growth:+.1f} MB")
            
            if abs(python_growth) < abs(rss_growth) * 0.1:
                print(f"\n✓ Python objects are stable")
                print(f"⚠ Memory leak is in native code (MPS/CUDA/FFmpeg)")
            else:
                print(f"\n⚠ Python objects are growing!")
                print(f"Top allocations:")
                for alloc in samples[-1].top_allocations[:5]:
                    print(f"  - {alloc['file']}: {alloc['size_mb']:.2f} MB")
            
            # Memory trend analysis
            if len(samples) >= 4:
                first_quarter = sum(s.rss_mb for s in samples[:len(samples)//4]) / (len(samples)//4)
                last_quarter = sum(s.rss_mb for s in samples[3*len(samples)//4:]) / (len(samples)//4)
                
                print(f"\nTrend Analysis:")
                print(f"  First quarter avg: {first_quarter:.1f} MB")
                print(f"  Last quarter avg:  {last_quarter:.1f} MB")
                
                if last_quarter > first_quarter * 1.2:
                    print(f"  ⚠ Memory growing steadily!")
                else:
                    print(f"  ✓ Memory stable")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = PROJECT_ROOT / f"memory_detailed_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "iterations": iterations,
            "delay_between": delay_between,
            "total_time_sec": total_elapsed,
            "samples": [
                {
                    "iteration": s.iteration,
                    "timestamp": s.timestamp,
                    "rss_mb": s.rss_mb,
                    "vms_mb": s.vms_mb,
                    "tracemalloc_current_mb": s.tracemalloc_current_mb,
                    "tracemalloc_peak_mb": s.tracemalloc_peak_mb,
                    "top_allocations": s.top_allocations,
                }
                for s in samples
            ],
        }
        
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {results_file}")
        
    finally:
        tracemalloc.stop()
        # Cleanup
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Detailed Memory Leak Analysis")
    parser.add_argument("--iterations", "-i", type=int, default=100)
    parser.add_argument("--delay", "-d", type=float, default=0.0,
                        help="Delay between requests in seconds (for MPS cleanup testing)")
    args = parser.parse_args()
    
    run_detailed_test(iterations=args.iterations, delay_between=args.delay)


if __name__ == "__main__":
    main()
