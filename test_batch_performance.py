#!/usr/bin/env python3
"""Test batch processing performance with actual measurements"""

import os
import time
import json

os.environ['NUMBA_DISABLE_JIT'] = '1'

import whisperx
from whisperx.process_separation import ProcessSeparatedPipeline

def measure_performance():
    """Measure actual performance of batch vs non-batch processing"""
    
    print("Batch Processing Performance Measurement")
    print("="*60)
    
    # Load test audio
    audio_file = "30m.wav"
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found")
        return
    
    print(f"Loading audio: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    audio_duration = len(audio) / 16000
    
    # Use first 2 minutes for faster testing
    test_duration = 120  # seconds
    test_audio = audio[:int(test_duration * 16000)]
    test_audio_duration = len(test_audio) / 16000
    
    print(f"Using first {test_audio_duration:.1f} seconds for testing")
    
    results = {}
    
    # Test 1: Sequential (batch_size=1)
    print(f"\n1. Sequential Processing (no batching):")
    pipeline_seq = ProcessSeparatedPipeline(
        asr_backend="mlx",
        model_name="tiny",
        vad_method="silero",
        device="mlx",
        language="en",
        compute_type="float16",
        asr_options={"word_timestamps": False},
        use_batch_processing=False,
        task="transcribe"
    )
    
    start = time.time()
    result_seq = pipeline_seq.transcribe(test_audio, verbose=False)
    seq_time = time.time() - start
    
    segments_seq = len(result_seq.get("segments", []))
    print(f"  Time: {seq_time:.2f}s")
    print(f"  Segments: {segments_seq}")
    print(f"  RTF: {test_audio_duration/seq_time:.1f}x")
    
    results['sequential'] = {
        'time': seq_time,
        'segments': segments_seq,
        'rtf': test_audio_duration/seq_time
    }
    
    # Test 2: Batch processing
    for batch_size in [4, 8]:
        print(f"\n2. Batch Processing (batch_size={batch_size}):")
        pipeline_batch = ProcessSeparatedPipeline(
            asr_backend="mlx",
            model_name="tiny",
            vad_method="silero",
            device="mlx",
            language="en",
            compute_type="float16",
            asr_options={"word_timestamps": False},
            use_batch_processing=True,
            batch_size=batch_size,
            task="transcribe"
        )
        
        start = time.time()
        result_batch = pipeline_batch.transcribe(test_audio, verbose=False)
        batch_time = time.time() - start
        
        segments_batch = len(result_batch.get("segments", []))
        print(f"  Time: {batch_time:.2f}s")
        print(f"  Segments: {segments_batch}")
        print(f"  RTF: {test_audio_duration/batch_time:.1f}x")
        
        results[f'batch_{batch_size}'] = {
            'time': batch_time,
            'segments': segments_batch,
            'rtf': test_audio_duration/batch_time
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Method':<20} {'Time (s)':<10} {'RTF':<10} {'vs Sequential':<15}")
    print("-" * 60)
    
    seq_time = results['sequential']['time']
    for method, data in results.items():
        speedup = seq_time / data['time'] if data['time'] > 0 else 0
        speedup_str = f"{speedup:.2f}x" if method != 'sequential' else "baseline"
        print(f"{method:<20} {data['time']:<10.2f} {data['rtf']:<10.1f} {speedup_str:<15}")
    
    # Save results
    output_file = "batch_performance_results.json"
    with open(output_file, "w") as f:
        json.dump({
            'test_duration': test_audio_duration,
            'results': results,
            'model': 'tiny'
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    measure_performance()