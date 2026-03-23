#!/usr/bin/env python3
"""
Детальный профайлинг памяти с tracemalloc - показывает точные строки кода.
"""

import os
import sys
import time
import tracemalloc
import tempfile
from pathlib import Path

import soundfile as sf
import requests

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample-data"
TEST_AUDIO_FILE = "Sobolev_Andrey_1_0_00-2_17.ogg"
API_ENDPOINT = "http://localhost:9007/v1/audio/transcriptions"
API_TOKEN = "key"

def get_process_memory_mb():
    """Get current process RSS memory."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    print("="*80)
    print("ДЕТАЛЬНЫЙ ПРОФАЙЛИНГ ПАМЯТИ С TRACEMALLOC")
    print("="*80)
    
    # Load test audio
    audio_path = SAMPLE_DATA_DIR / TEST_AUDIO_FILE
    audio_data, sample_rate = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    print(f"Loaded audio: {len(audio_data)} samples")
    
    # Start tracemalloc
    tracemalloc.start()
    print("\n✓ tracemalloc запущен")
    
    # Baseline snapshot
    baseline = tracemalloc.take_snapshot()
    baseline_mem = get_process_memory_mb()
    print(f"Baseline memory: {baseline_mem:.1f} MB")
    print(f"Baseline Python objects: {tracemalloc.get_traced_memory()[0] / 1024**2:.2f} MB")
    
    # Run 50 requests with detailed profiling
    print("\n" + "="*80)
    print("Запуск 50 транскрипций с профайлингом...")
    print("="*80)
    
    top_allocations = []
    
    for i in range(50):
        # Extract 10-20s segment
        import numpy as np
        import random
        
        duration = random.uniform(10, 20)
        start = random.uniform(0, max(0, len(audio_data)/sample_rate - duration))
        start_sample = int(start * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        segment = audio_data[start_sample:end_sample].astype(np.float32)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, segment, sample_rate)
            temp_path = f.name
        
        # Take snapshot before request
        snap_before = tracemalloc.take_snapshot()
        
        # Send request
        with open(temp_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            data = {'model': 'whisper-1', 'language': 'ru', 'response_format': 'verbose_json'}
            response = requests.post(
                API_ENDPOINT,
                headers={'Authorization': f'Bearer {API_TOKEN}'},
                files=files,
                data=data,
                timeout=60
            )
        
        os.unlink(temp_path)
        
        # Take snapshot after request
        snap_after = tracemalloc.take_snapshot()
        
        # Calculate diff
        diff = snap_after.compare_to(snap_before, 'lineno')
        
        # Get top allocations from our code
        for stat in diff[:5]:
            filename = stat.traceback.format()[-1] if stat.traceback.format() else "unknown"
            # Filter for our code
            if any(x in filename for x in ['src/', 'vendor/gigaam/', 'main.py']):
                size_mb = stat.size_diff / (1024 * 1024)
                if size_mb > 0.1:  # Only significant allocations
                    top_allocations.append({
                        'iteration': i + 1,
                        'file': filename,
                        'size_mb': size_mb,
                        'count': stat.count_diff
                    })
        
        if (i + 1) % 10 == 0:
            current_mem = get_process_memory_mb()
            python_mem = tracemalloc.get_traced_memory()[0] / 1024**2
            print(f"[{i+1}/50] RSS: {current_mem:.0f}MB | Python: {python_mem:.2f}MB")
    
    # Final analysis
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    final = tracemalloc.take_snapshot()
    total_diff = final.compare_to(baseline, 'lineno')
    
    final_mem = get_process_memory_mb()
    final_python_mem = tracemalloc.get_traced_memory()[0] / 1024**2
    
    print(f"\nProcess Memory (RSS):")
    print(f"  Start: {baseline_mem:.1f} MB")
    print(f"  End:   {final_mem:.1f} MB")
    print(f"  Delta: {final_mem - baseline_mem:+.1f} MB")
    
    print(f"\nPython Objects (tracemalloc):")
    print(f"  Start: 0.00 MB")
    print(f"  End:   {final_python_mem:.2f} MB")
    print(f"  Delta: {final_python_mem:+.2f} MB")
    
    rss_leak = final_mem - baseline_mem
    python_leak = final_python_mem
    native_leak = rss_leak - python_leak
    
    print(f"\n=== КЛЮЧЕВЫЕ ВЫВОДЫ ===")
    print(f"RSS утечка:       {rss_leak:+.1f} MB")
    print(f"Python утечка:    {python_leak:+.2f} MB")
    print(f"Native/MPS утечка: {native_leak:+.1f} MB")
    
    if python_leak < rss_leak * 0.1:
        print(f"\n✓ Python объекты НЕ растут")
        print(f"⚠ Утечка в native коде (MPS/CUDA/librosa/soundfile)")
    else:
        print(f"\n⚠ Python объекты РАСТУТ!")
        print(f"\nТоп Python allocations:")
        for stat in total_diff[:10]:
            filename = stat.traceback.format()[-1] if stat.traceback.format() else "unknown"
            if any(x in filename for x in ['src/', 'vendor/']):
                print(f"  {filename}")
                print(f"    Size: {stat.size_diff / 1024**2:+.2f} MB, Count: {stat.count_diff:+d}")
    
    # Show top allocations during test
    if top_allocations:
        print(f"\n=== ТОП ALLOCATION МОМЕНТЫ ===")
        # Group by file
        from collections import defaultdict
        file_totals = defaultdict(float)
        for alloc in top_allocations:
            file_totals[alloc['file']] += alloc['size_mb']
        
        sorted_files = sorted(file_totals.items(), key=lambda x: x[1], reverse=True)
        for file, total_mb in sorted_files[:5]:
            print(f"\n{file}")
            print(f"  Total: {total_mb:.2f} MB")
    
    tracemalloc.stop()
    print("\n✓ Профайлинг завершен")

if __name__ == "__main__":
    main()
