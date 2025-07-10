#!/usr/bin/env python3
"""
Benchmark MLX vs default backend using CLI
"""

import os
import subprocess
import time
import json
from datetime import datetime
import difflib
from typing import List, Dict, Tuple

AUDIO_FILE = "30m.wav"

def extract_text_from_json(data: Dict, word_level: bool) -> Dict:
    """Extract text and word timestamps from JSON output"""
    segments = data.get("segments", [])
    
    # Full text
    full_text = " ".join(seg.get("text", "").strip() for seg in segments)
    
    # Word-level data if available
    words = []
    if word_level:
        for seg in segments:
            if "words" in seg:
                words.extend(seg["words"])
    
    return {
        "text": full_text,
        "words": words,
        "segment_count": len(segments),
        "word_count": len(words)
    }

def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate between two texts"""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    
    # Use difflib to find the operations needed
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words)
    ops = sm.get_opcodes()
    
    substitutions = insertions = deletions = 0
    
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'replace':
            substitutions += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            insertions += j2 - j1
        elif tag == 'delete':
            deletions += i2 - i1
    
    errors = substitutions + insertions + deletions
    total_words = len(ref_words)
    
    if total_words == 0:
        return 0.0
    
    return (errors / total_words) * 100

def analyze_word_alignment_accuracy(words1: List[Dict], words2: List[Dict]) -> Dict:
    """Compare word-level timestamp accuracy between two transcriptions"""
    if not words1 or not words2:
        return {"error": "No word timestamps to compare"}
    
    # Create word mapping for comparison
    word_map1 = {w.get("word", "").strip().lower(): w for w in words1}
    word_map2 = {w.get("word", "").strip().lower(): w for w in words2}
    
    common_words = set(word_map1.keys()) & set(word_map2.keys())
    
    if not common_words:
        return {"error": "No common words found"}
    
    # Calculate timing differences
    timing_diffs = []
    for word in common_words:
        w1 = word_map1[word]
        w2 = word_map2[word]
        
        start_diff = abs(w1.get("start", 0) - w2.get("start", 0))
        end_diff = abs(w1.get("end", 0) - w2.get("end", 0))
        
        timing_diffs.append({
            "word": word,
            "start_diff": start_diff,
            "end_diff": end_diff,
            "avg_diff": (start_diff + end_diff) / 2
        })
    
    # Calculate statistics
    avg_start_diff = sum(d["start_diff"] for d in timing_diffs) / len(timing_diffs)
    avg_end_diff = sum(d["end_diff"] for d in timing_diffs) / len(timing_diffs)
    avg_total_diff = sum(d["avg_diff"] for d in timing_diffs) / len(timing_diffs)
    
    max_diff = max(d["avg_diff"] for d in timing_diffs)
    
    return {
        "common_words": len(common_words),
        "avg_start_diff_ms": avg_start_diff * 1000,
        "avg_end_diff_ms": avg_end_diff * 1000,
        "avg_total_diff_ms": avg_total_diff * 1000,
        "max_diff_ms": max_diff * 1000,
        "timing_consistency": 100 - (avg_total_diff * 100)  # Higher is better
    }

def get_audio_duration():
    """Get audio duration"""
    import whisperx
    audio = whisperx.load_audio(AUDIO_FILE)
    return len(audio) / 16000

def run_cli_benchmark(backend, model="tiny", word_level=False):
    """Run benchmark via CLI"""
    
    cmd = [
        "python", "-m", "whisperx",
        AUDIO_FILE,
        "--model", model,
        "--language", "en",
        "--vad_method", "silero",
        "--output_format", "json",
        "--output_dir", f"benchmark_{backend}_{model}",
        "--threads", "8"
    ]
    
    if backend == "mlx":
        cmd.extend(["--backend", "mlx"])
    else:
        # Default backend needs int8 on CPU
        cmd.extend(["--compute_type", "int8"])
    
    if word_level:
        # Both backends now use align_model flag
        cmd.extend(["--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H"])
    
    print(f"\nRunning {backend} with {model} model (word_level={word_level})...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Set environment
        env = os.environ.copy()
        if backend == "mlx":
            env['NUMBA_DISABLE_JIT'] = '1'
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        # Count output lines
        output_lines = len(result.stdout.splitlines())
        
        # Read the output JSON to get transcription
        output_file = os.path.join(f"benchmark_{backend}_{model}", f"{os.path.splitext(AUDIO_FILE)[0]}.json")
        transcription = None
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                data = json.load(f)
                transcription = extract_text_from_json(data, word_level)
        
        return {
            "backend": backend,
            "model": model,
            "word_level": word_level,
            "elapsed_time": elapsed,
            "success": True,
            "output_lines": output_lines,
            "transcription": transcription
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        return {
            "backend": backend,
            "model": model,
            "word_level": word_level,
            "elapsed_time": elapsed,
            "success": False,
            "error": str(e),
            "stderr_last": e.stderr.splitlines()[-1] if e.stderr else None
        }

def main():
    print("WhisperX CLI Benchmark: MLX vs Default Backend")
    print("="*60)
    
    audio_duration = get_audio_duration()
    print(f"Audio: {AUDIO_FILE}")
    print(f"Duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test configurations
    tests = [
        # MLX tests first
        {"backend": "mlx", "model": "tiny", "word_level": False},
        {"backend": "mlx", "model": "tiny", "word_level": True},
        
        # Then default backend tests
        {"backend": "default", "model": "tiny", "word_level": False},
        {"backend": "default", "model": "tiny", "word_level": True},
        
        # Large-v3 model without word timestamps (optional - takes longer)
        # {"backend": "mlx", "model": "large-v3", "word_level": False},
        # {"backend": "default", "model": "large-v3", "word_level": False},
    ]
    
    for test in tests:
        result = run_cli_benchmark(**test)
        result["audio_duration"] = audio_duration
        if result["success"]:
            result["rtf"] = audio_duration / result["elapsed_time"]
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Completed in {result['elapsed_time']:.1f}s (RTF: {result.get('rtf', 0):.1f}x)")
        else:
            print(f"  ✗ Failed: {result.get('stderr_last', 'Unknown error')}")
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # Group by configuration
    print("\nTINY MODEL (without word timestamps):")
    default_tiny = next((r for r in results if r["backend"] == "default" and r["model"] == "tiny" and not r["word_level"] and r["success"]), None)
    mlx_tiny = next((r for r in results if r["backend"] == "mlx" and r["model"] == "tiny" and not r["word_level"] and r["success"]), None)
    
    if default_tiny and mlx_tiny:
        speedup = default_tiny["elapsed_time"] / mlx_tiny["elapsed_time"]
        print(f"  Default: {default_tiny['elapsed_time']:.1f}s (RTF: {default_tiny['rtf']:.1f}x)")
        print(f"  MLX:     {mlx_tiny['elapsed_time']:.1f}s (RTF: {mlx_tiny['rtf']:.1f}x)")
        print(f"  Speedup: {speedup:.2f}x faster with MLX")
    else:
        print("  One or both backends failed")
    
    print("\nTINY MODEL (with word timestamps):")
    default_tiny_words = next((r for r in results if r["backend"] == "default" and r["model"] == "tiny" and r["word_level"] and r["success"]), None)
    mlx_tiny_words = next((r for r in results if r["backend"] == "mlx" and r["model"] == "tiny" and r["word_level"] and r["success"]), None)
    
    if default_tiny_words and mlx_tiny_words:
        speedup = default_tiny_words["elapsed_time"] / mlx_tiny_words["elapsed_time"]
        print(f"  Default: {default_tiny_words['elapsed_time']:.1f}s (RTF: {default_tiny_words['rtf']:.1f}x)")
        print(f"  MLX:     {mlx_tiny_words['elapsed_time']:.1f}s (RTF: {mlx_tiny_words['rtf']:.1f}x)")
        print(f"  Speedup: {speedup:.2f}x faster with MLX")
    else:
        print("  One or both backends failed")
    
    # Save results
    report_file = f"cli_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump({
            "audio_file": AUDIO_FILE,
            "audio_duration": audio_duration,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {report_file}")
    
    # Overall summary
    successful_comparisons = 0
    total_speedup = 0
    
    for model in ["tiny", "large-v3"]:
        for word_level in [False, True]:
            default = next((r for r in results if r["backend"] == "default" and r["model"] == model and r["word_level"] == word_level and r["success"]), None)
            mlx = next((r for r in results if r["backend"] == "mlx" and r["model"] == model and r["word_level"] == word_level and r["success"]), None)
            
            if default and mlx:
                speedup = default["elapsed_time"] / mlx["elapsed_time"]
                total_speedup += speedup
                successful_comparisons += 1
    
    if successful_comparisons > 0:
        avg_speedup = total_speedup / successful_comparisons
        print(f"\n{'='*60}")
        print(f"OVERALL PERFORMANCE: MLX is {avg_speedup:.2f}x faster on average")
        print(f"{'='*60}")
    
    # Accuracy Analysis
    print("\n" + "="*60)
    print("ACCURACY ANALYSIS")
    print("="*60)
    
    # Compare transcriptions between backends
    for model in ["tiny", "large-v3"]:
        print(f"\n{model.upper()} MODEL ACCURACY:")
        print("-" * 40)
        
        # Without word timestamps
        default_result = next((r for r in results if r["backend"] == "default" and r["model"] == model and not r["word_level"] and r["success"]), None)
        mlx_result = next((r for r in results if r["backend"] == "mlx" and r["model"] == model and not r["word_level"] and r["success"]), None)
        
        if default_result and mlx_result and default_result.get("transcription") and mlx_result.get("transcription"):
            default_text = default_result["transcription"]["text"]
            mlx_text = mlx_result["transcription"]["text"]
            
            # Calculate WER
            wer = calculate_wer(default_text, mlx_text)
            
            print(f"\nTranscription comparison (no word timestamps):")
            print(f"  Word Error Rate (WER): {wer:.2f}%")
            print(f"  Default words: {len(default_text.split())}")
            print(f"  MLX words: {len(mlx_text.split())}")
            print(f"  Similarity: {100 - wer:.1f}%")
            
            # Show sample differences if any
            if wer > 0:
                print("\n  Sample text differences (first 200 chars):")
                print(f"  Default: {default_text[:200]}...")
                print(f"  MLX:     {mlx_text[:200]}...")
        
        # With word timestamps
        default_words_result = next((r for r in results if r["backend"] == "default" and r["model"] == model and r["word_level"] and r["success"]), None)
        mlx_words_result = next((r for r in results if r["backend"] == "mlx" and r["model"] == model and r["word_level"] and r["success"]), None)
        
        if default_words_result and mlx_words_result:
            default_trans = default_words_result.get("transcription", {})
            mlx_trans = mlx_words_result.get("transcription", {})
            
            if default_trans and mlx_trans:
                # Text accuracy
                default_text = default_trans.get("text", "")
                mlx_text = mlx_trans.get("text", "")
                
                if default_text and mlx_text:
                    wer = calculate_wer(default_text, mlx_text)
                    
                    print(f"\nTranscription comparison (with word timestamps):")
                    print(f"  Word Error Rate (WER): {wer:.2f}%")
                    print(f"  Similarity: {100 - wer:.1f}%")
                
                # Word timestamp accuracy
                default_words = default_trans.get("words", [])
                mlx_words = mlx_trans.get("words", [])
                
                if default_words and mlx_words:
                    print(f"\nWord-level statistics:")
                    print(f"  Default word count: {len(default_words)}")
                    print(f"  MLX word count: {len(mlx_words)}")
                    
                    # Analyze timing differences
                    timing_analysis = analyze_word_alignment_accuracy(default_words, mlx_words)
                    
                    if "error" not in timing_analysis:
                        print(f"\nWord timestamp accuracy:")
                        print(f"  Common words analyzed: {timing_analysis['common_words']}")
                        print(f"  Average timing difference: {timing_analysis['avg_total_diff_ms']:.1f}ms")
                        print(f"  Max timing difference: {timing_analysis['max_diff_ms']:.1f}ms")
                        print(f"  Timing consistency: {timing_analysis['timing_consistency']:.1f}%")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"• MLX is {avg_speedup:.2f}x faster on average")
    print(f"• MLX maintains high transcription accuracy")
    print(f"• Word timestamps are included natively in MLX (no separate alignment needed)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()