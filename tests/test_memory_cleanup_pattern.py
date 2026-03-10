#!/usr/bin/env python3
"""
Quick memory leak test for GigaAM without loading the full model.
This demonstrates the memory cleanup patterns we've implemented.
"""

import gc
import numpy as np
import torch

# Mock the GigaAM model for testing
class MockGigaAM:
    """Mock GigaAM model for memory testing."""
    
    def __init__(self):
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        
    class MockDecoding:
        def decode(self, head, encoded, encoded_len):
            return ["test transcription"]
    
    class MockHead:
        pass
    
    def __init__(self):
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.decoding = self.MockDecoding()
        self.head = self.MockHead()
    
    def forward(self, wav, length):
        # Simulate memory allocation
        encoded = torch.randn(1, 100, 512, device=self._device, dtype=self._dtype)
        encoded_len = length
        return encoded, encoded_len


def test_memory_cleanup_pattern():
    """Test that our memory cleanup pattern works correctly."""
    print("Testing memory cleanup pattern...")
    
    model = MockGigaAM()
    
    def get_memory_mb():
        gc.collect()
        return torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    
    # Simulate transcription with cleanup
    audio = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds
    wav_tensor = torch.from_numpy(audio)
    device = model._device
    dtype = model._dtype
    
    wav = wav_tensor.to(device).to(dtype).unsqueeze(0)
    length = torch.full([1], wav.shape[-1], device=device)
    
    encoded = None
    encoded_len = None
    
    try:
        encoded, encoded_len = model.forward(wav, length)
        result = model.decoding.decode(model.head, encoded, encoded_len)[0]
        print(f"Transcription result: {result}")
    finally:
        # Our cleanup pattern
        del wav, length
        if encoded is not None:
            del encoded
        if encoded_len is not None:
            del encoded_len
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    print("✓ Memory cleanup pattern executed successfully")
    
    # Verify tensors are deleted
    try:
        _ = wav
        print("✗ ERROR: wav tensor still exists!")
    except NameError:
        print("✓ wav tensor properly deleted")
    
    try:
        _ = encoded
        print("✗ ERROR: encoded tensor still exists!")
    except NameError:
        print("✓ encoded tensor properly deleted")
    
    print("\nMemory cleanup test completed!")


if __name__ == "__main__":
    test_memory_cleanup_pattern()
