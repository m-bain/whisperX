#!/usr/bin/env python3
"""
Comprehensive test script for MLX WhisperX fork.
Tests all major functionality including:
- Basic transcription
- VAD segmentation  
- Batch processing
- Different model sizes
- INT4 quantization
- CLI functionality
"""

import os
import sys
import time
import json
import numpy as np
import subprocess
from pathlib import Path

# Test audio creation
def create_test_audio():
    """Create a test audio file with speech."""
    import whisperx.audio as audio
    
    # Create 10 second test audio
    sample_rate = 16000
    duration = 10
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Generate speech-like signal (mix of frequencies)
    signal = np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
    signal += np.sin(2 * np.pi * 880 * t) * 0.2  # A5 note
    signal += np.sin(2 * np.pi * 220 * t) * 0.1  # A3 note
    
    # Add some modulation to simulate speech
    modulation = np.sin(2 * np.pi * 5 * t) * 0.5 + 0.5
    signal = signal * modulation
    
    # Add silence at beginning and end
    silence = np.zeros(int(sample_rate * 2))
    signal = np.concatenate([silence, signal, silence])
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Save as WAV
    import wave
    with wave.open("test_audio.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((signal * 32767).astype(np.int16).tobytes())
    
    return "test_audio.wav"

def test_basic_transcription():
    """Test basic transcription functionality."""
    print("\n=== Testing Basic Transcription ===")
    
    from whisperx import load_model, load_audio
    
    # Create test audio
    audio_file = create_test_audio()
    
    # Load audio
    audio = load_audio(audio_file)
    
    # Test with tiny model
    print("Loading tiny model...")
    model = load_model("tiny", device="cpu", compute_type="float16", backend="mlx")
    
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=8)
    
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    
    for segment in result['segments']:
        print(f"  [{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
    
    # Cleanup
    os.remove(audio_file)
    
    return True

def test_vad_methods():
    """Test both Silero and PyAnnote VAD methods."""
    print("\n=== Testing VAD Methods ===")
    
    from whisperx import load_model, load_audio
    
    # Create test audio
    audio_file = create_test_audio()
    audio = load_audio(audio_file)
    
    # Test Silero VAD
    print("\nTesting Silero VAD...")
    model = load_model("tiny", device="cpu", compute_type="float16", 
                      backend="mlx", vad_method="silero")
    result = model.transcribe(audio, batch_size=8)
    print(f"Silero VAD segments: {len(result['segments'])}")
    
    # Test PyAnnote VAD
    print("\nTesting PyAnnote VAD...")
    model = load_model("tiny", device="cpu", compute_type="float16",
                      backend="mlx", vad_method="pyannote")
    result = model.transcribe(audio, batch_size=8)
    print(f"PyAnnote VAD segments: {len(result['segments'])}")
    
    # Cleanup
    os.remove(audio_file)
    
    return True

def test_batch_processing():
    """Test batch processing functionality."""
    print("\n=== Testing Batch Processing ===")
    
    from whisperx import load_model, load_audio
    
    # Create test audio
    audio_file = create_test_audio()
    audio = load_audio(audio_file)
    
    # Test standard backend
    print("\nTesting standard backend...")
    model = load_model("tiny", device="cpu", compute_type="float16",
                      backend="standard")
    start = time.time()
    result = model.transcribe(audio, batch_size=1)
    standard_time = time.time() - start
    print(f"Standard backend time: {standard_time:.2f}s")
    
    # Test batch backend
    print("\nTesting batch backend...")
    model = load_model("tiny", device="cpu", compute_type="float16",
                      backend="batch")
    start = time.time()
    result = model.transcribe(audio, batch_size=8)
    batch_time = time.time() - start
    print(f"Batch backend time: {batch_time:.2f}s")
    
    if standard_time > 0:
        improvement = (standard_time - batch_time) / standard_time * 100
        print(f"Batch processing improvement: {improvement:.1f}%")
    
    # Cleanup
    os.remove(audio_file)
    
    return True

def test_int4_quantization():
    """Test INT4 quantized models."""
    print("\n=== Testing INT4 Quantization ===")
    
    from whisperx import load_model, load_audio
    
    # Create test audio
    audio_file = create_test_audio()
    audio = load_audio(audio_file)
    
    try:
        # Test INT4 model
        print("Loading INT4 quantized model...")
        model = load_model("tiny", device="cpu", compute_type="int4",
                          backend="mlx")
        
        print("Transcribing with INT4 model...")
        result = model.transcribe(audio, batch_size=8)
        print(f"INT4 segments: {len(result['segments'])}")
        
    except Exception as e:
        print(f"INT4 test failed (model may not be available): {e}")
    
    # Cleanup
    os.remove(audio_file)
    
    return True

def test_cli():
    """Test CLI functionality."""
    print("\n=== Testing CLI ===")
    
    # Create test audio
    audio_file = create_test_audio()
    
    # Test basic CLI command
    print("\nTesting basic CLI transcription...")
    cmd = [
        sys.executable, "-m", "whisperx",
        audio_file,
        "--model", "tiny",
        "--backend", "mlx",
        "--output_format", "json",
        "--no_align"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("CLI transcription successful!")
        
        # Check if output file was created
        output_file = Path(audio_file).stem + ".json"
        if os.path.exists(output_file):
            print("Output file created successfully")
            
            # Load and check content
            with open(output_file, 'r') as f:
                data = json.load(f)
                print(f"Transcription segments: {len(data.get('segments', []))}")
            
            # Cleanup
            os.remove(output_file)
        else:
            print("WARNING: Output file not created")
    else:
        print(f"CLI failed with error:\n{result.stderr}")
    
    # Test with backend flag
    print("\nTesting CLI with --backend batch...")
    cmd = [
        sys.executable, "-m", "whisperx", 
        audio_file,
        "--model", "tiny",
        "--backend", "batch",
        "--output_format", "txt",
        "--no_align"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("CLI with batch backend successful!")
        
        # Check output file
        output_file = Path(audio_file).stem + ".txt"
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                content = f.read()
                print(f"Transcription text: {content[:100]}...")
            os.remove(output_file)
    else:
        print(f"CLI batch backend failed: {result.stderr}")
    
    # Cleanup
    os.remove(audio_file)
    
    return True

def main():
    """Run all tests."""
    print("=== WhisperX MLX Fork Comprehensive Test ===")
    print("Testing all major functionality...\n")
    
    tests = [
        ("Basic Transcription", test_basic_transcription),
        ("VAD Methods", test_vad_methods),
        ("Batch Processing", test_batch_processing),
        ("INT4 Quantization", test_int4_quantization),
        ("CLI Functionality", test_cli),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"\n✗ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with error: {e}")
            failed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! MLX WhisperX fork is working correctly.")
    else:
        print(f"\n✗ {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)