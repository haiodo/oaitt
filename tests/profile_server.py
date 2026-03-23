#!/usr/bin/env python3
"""
Профайлинг на стороне сервера через API.
Добавляет endpoint /admin/profile-memory для анализа памяти.
"""

import requests
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import os

API_ENDPOINT = "http://localhost:9007/v1/audio/transcriptions"
ADMIN_ENDPOINT = "http://localhost:9007/admin/profile-memory"

def main():
    print("Профайлинг памяти на сервере...")
    
    # Load audio
    audio_path = Path('sample-data/Sobolev_Andrey_1_0_00-2_17.ogg')
    audio_data, sample_rate = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Start profiling on server
    requests.post(ADMIN_ENDPOINT, json={'action': 'start'})
    print("✓ Профайлинг на сервере запущен")
    
    # Send 20 requests
    for i in range(20):
        # Random segment
        duration = np.random.uniform(10, 20)
        start = np.random.uniform(0, max(0, len(audio_data)/sample_rate - duration))
        start_sample = int(start * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        segment = audio_data[start_sample:end_sample].astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, segment, sample_rate)
            temp_path = f.name
        
        with open(temp_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            data = {'model': 'whisper-1', 'language': 'ru', 'response_format': 'verbose_json'}
            response = requests.post(API_ENDPOINT, 
                                    headers={'Authorization': 'Bearer key'},
                                    files=files, data=data, timeout=60)
        
        os.unlink(temp_path)
        
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/20] Requests sent")
    
    # Get results
    result = requests.post(ADMIN_ENDPOINT, json={'action': 'stop'}).json()
    
    print("\n=== РЕЗУЛЬТАТЫ СЕРВЕРНОГО ПРОФАЙЛИНГА ===")
    print(f"Python memory growth: {result.get('python_growth_mb', 0):.2f} MB")
    print(f"RSS growth: {result.get('rss_growth_mb', 0):.2f} MB")
    
    if 'top_allocations' in result:
        print("\nТоп allocations:")
        for alloc in result['top_allocations'][:5]:
            print(f"  {alloc['file']}: {alloc['size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
