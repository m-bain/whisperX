#!/usr/bin/env python3
"""
Benchmark different MLX models and compare performance
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import whisperx


def benchmark_model(audio_file: str, model_name: str, num_runs: int = 3) -> Dict:
    """
    Benchmark a single model.
    
    Args:
        audio_file: Path to audio file
        model_name: Model name (e.g., "tiny", "base", "large-v3")
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking {model_name}...")
    
    # Load audio once
    audio = whisperx.load_audio(audio_file)
    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.1f}s")
    
    # Load model
    print("Loading model...")
    start = time.time()
    model = whisperx.load_model(
        model_name,
        device="mlx",
        compute_type="float16" if not model_name.endswith("-q4") else "int4",
        asr_backend="mlx"
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Warm-up run
    print("Warm-up run...")
    _ = model.transcribe(audio[:16000], language="en")  # 1 second warm-up
    
    # Benchmark runs
    times = []
    results = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        start = time.time()
        result = model.transcribe(
            audio,
            language="en",
            print_progress=False
        )
        elapsed = time.time() - start
        times.append(elapsed)
        results.append(result)
        print(f"  Time: {elapsed:.2f}s, RTF: {duration/elapsed:.1f}x")
    
    # Calculate statistics
    times_array = np.array(times)
    
    benchmark_result = {
        "model": model_name,
        "audio_duration": duration,
        "load_time": load_time,
        "times": times,
        "mean_time": float(np.mean(times_array)),
        "std_time": float(np.std(times_array)),
        "min_time": float(np.min(times_array)),
        "max_time": float(np.max(times_array)),
        "mean_rtf": duration / np.mean(times_array),
        "transcription": results[0]["text"][:200] + "...",
        "num_segments": len(results[0]["segments"]),
        "detected_language": results[0].get("language", "unknown")
    }
    
    # Memory usage estimate (rough)
    if model_name.endswith("-q4"):
        model_size_map = {
            "tiny-q4": 0.02,
            "base-q4": 0.04,
            "small-q4": 0.12,
            "large-v3-q4": 0.75
        }
    else:
        model_size_map = {
            "tiny": 0.04,
            "base": 0.07,
            "small": 0.24,
            "medium": 0.77,
            "large-v3": 1.55
        }
    
    benchmark_result["model_size_gb"] = model_size_map.get(model_name, 0)
    
    return benchmark_result


def compare_models(audio_file: str, models: List[str], num_runs: int = 3):
    """
    Compare multiple models.
    
    Args:
        audio_file: Audio file to benchmark
        models: List of model names
        num_runs: Number of runs per model
    """
    print(f"Comparing {len(models)} models on: {audio_file}")
    print(f"Number of runs per model: {num_runs}")
    
    results = []
    
    for model_name in models:
        try:
            result = benchmark_model(audio_file, model_name, num_runs)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Size':>8} {'Load':>8} {'Mean Time':>10} {'RTF':>8} {'Std Dev':>8}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x["mean_rtf"], reverse=True):
        print(f"{r['model']:<15} "
              f"{r['model_size_gb']:>6.2f}GB "
              f"{r['load_time']:>6.2f}s "
              f"{r['mean_time']:>8.2f}s "
              f"{r['mean_rtf']:>6.1f}x "
              f"{r['std_time']:>6.2f}s")
    
    # Quality comparison (simplified - just text length)
    print("\n" + "="*80)
    print("TRANSCRIPTION PREVIEW")
    print("="*80)
    
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  {r['transcription']}")
    
    # Save detailed results
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "audio_file": audio_file,
            "audio_duration": results[0]["audio_duration"] if results else 0,
            "num_runs": num_runs,
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Performance/quality recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if results:
        fastest = max(results, key=lambda x: x["mean_rtf"])
        most_accurate = max(results, key=lambda x: len(x["transcription"]))
        
        print(f"Fastest model: {fastest['model']} ({fastest['mean_rtf']:.1f}x real-time)")
        print(f"Most detailed: {most_accurate['model']}")
        
        # Find best balance (RTF > 30 with largest size)
        balanced = [r for r in results if r["mean_rtf"] > 30]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x["model_size_gb"])
            print(f"Best balanced: {best_balanced['model']} "
                  f"({best_balanced['mean_rtf']:.1f}x RT, "
                  f"{best_balanced['model_size_gb']:.2f}GB)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <audio_file> [models...]")
        print("\nExamples:")
        print("  python benchmark.py audio.mp3")
        print("  python benchmark.py audio.mp3 tiny base large-v3")
        print("  python benchmark.py audio.mp3 tiny tiny-q4 base base-q4")
        print("\nDefault models: tiny, base, small, large-v3, and their INT4 variants")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        models = sys.argv[2:]
    else:
        # Default model comparison
        models = [
            "tiny",
            "tiny-q4",
            "base",
            "base-q4",
            "small",
            "small-q4",
            "large-v3",
            "large-v3-q4"
        ]
    
    try:
        compare_models(audio_file, models, num_runs=3)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()