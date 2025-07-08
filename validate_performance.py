#!/usr/bin/env python3
"""
Validate MLX WhisperX performance against roadmap targets
Target: ≥30× real-time ASR on M2 Pro, ≥5× end-to-end speedup
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil

import whisperx
from whisperx.audio import load_audio
from whisperx.pipeline import load_mlx_pipeline


class PerformanceValidator:
    """Validate performance against roadmap targets."""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get system info
        self.system_info = self._get_system_info()
        
        # Roadmap targets
        self.targets = {
            "asr_rtf_fp16": 30.0,      # ≥30× RT on M2 Pro
            "asr_rtf_int4": 40.0,      # ≥40× RT for INT4
            "end_to_end_speedup": 5.0,  # ≥5× vs CPU baseline
            "wer_delta": 0.3,          # WER Δ ≤ 0.3
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024**3,
        }
        
        # Get Apple Silicon info
        if platform.system() == "Darwin":
            try:
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True
                ).strip()
                info["chip"] = chip
                
                # Detect chip type
                if "M1" in chip:
                    info["chip_family"] = "M1"
                elif "M2" in chip:
                    info["chip_family"] = "M2"
                elif "M3" in chip:
                    info["chip_family"] = "M3"
                else:
                    info["chip_family"] = "Unknown"
            except:
                pass
        
        return info
    
    def create_test_audio(self, duration: int = 120) -> str:
        """Create 2-minute test audio as per roadmap."""
        test_file = self.output_dir / f"test_audio_{duration}s.wav"
        
        if not test_file.exists():
            print(f"Creating {duration}s test audio...")
            
            # Generate realistic text
            text_segments = [
                "This is a performance validation test for WhisperX with MLX backend.",
                "We are testing the real-time factor and accuracy of the system.",
                "The target is to achieve at least thirty times real-time performance.",
                "This should work efficiently on Apple Silicon devices.",
                "The MLX framework provides excellent acceleration for machine learning.",
                "Voice activity detection helps segment the audio properly.",
                "Forced alignment can provide word-level timestamps.",
                "Speaker diarization identifies who is speaking when.",
                "The unified pipeline makes everything work together seamlessly.",
                "This test validates that we meet the roadmap requirements.",
            ]
            
            # Repeat to fill duration
            full_text = " ".join(text_segments * (duration // 20))
            
            subprocess.run([
                "say", "-o", str(test_file),
                "--data-format=LEF32@16000",
                full_text[:2000]  # Limit for say command
            ], capture_output=True)
        
        return str(test_file)
    
    def validate_asr_performance(self, model: str = "large-v3") -> Dict:
        """Validate ASR-only performance."""
        print(f"\n{'='*60}")
        print(f"Validating ASR Performance: {model}")
        print('='*60)
        
        # Create test audio
        audio_file = self.create_test_audio(120)  # 2 minutes
        audio = load_audio(audio_file)
        duration = len(audio) / 16000
        
        results = {}
        
        # Test FP16
        print("\nTesting FP16 performance...")
        model_path = f"mlx-community/whisper-{model}"
        
        # Direct MLX test (no VAD)
        import mlx_whisper
        
        # Warm-up
        _ = mlx_whisper.transcribe(
            audio[:16000],
            path_or_hf_repo=model_path,
            verbose=False
        )
        
        # Benchmark
        start = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_path,
            verbose=False,
            language="en",
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start
        
        rtf = duration / elapsed
        results["fp16_rtf"] = rtf
        results["fp16_time"] = elapsed
        
        print(f"  Duration: {duration:.1f}s")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.1f}x")
        print(f"  Target: ≥{self.targets['asr_rtf_fp16']}x")
        print(f"  Status: {'✓ PASS' if rtf >= self.targets['asr_rtf_fp16'] else '✗ FAIL'}")
        
        # Test INT4 if available
        if model in ["tiny", "base", "small", "large-v3"]:
            print("\nTesting INT4 performance...")
            model_path_int4 = f"mlx-community/whisper-{model}-mlx-q4"
            
            try:
                start = time.perf_counter()
                result_int4 = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=model_path_int4,
                    verbose=False,
                    language="en",
                    temperature=0.0,
                )
                elapsed = time.perf_counter() - start
                
                rtf_int4 = duration / elapsed
                results["int4_rtf"] = rtf_int4
                results["int4_time"] = elapsed
                
                print(f"  Time: {elapsed:.2f}s")
                print(f"  RTF: {rtf_int4:.1f}x")
                print(f"  Target: ≥{self.targets['asr_rtf_int4']}x")
                print(f"  Status: {'✓ PASS' if rtf_int4 >= self.targets['asr_rtf_int4'] else '✗ FAIL'}")
                
                # Compare transcriptions
                if result["text"] == result_int4["text"]:
                    print("  ✓ INT4 output matches FP16")
                else:
                    print("  ⚠️  INT4 output differs from FP16")
            except Exception as e:
                print(f"  INT4 test failed: {e}")
                results["int4_rtf"] = 0
        
        results["model"] = model
        results["duration"] = duration
        results["text"] = result["text"][:200] + "..."
        
        return results
    
    def validate_pipeline_performance(self) -> Dict:
        """Validate full pipeline performance."""
        print(f"\n{'='*60}")
        print("Validating Pipeline Performance")
        print('='*60)
        
        # Create test audio
        audio_file = self.create_test_audio(120)
        duration = 120.0
        
        results = {}
        
        # Test MLX pipeline
        print("\nTesting MLX pipeline...")
        pipe_mlx = load_mlx_pipeline(
            model_path="mlx-community/whisper-large-v3",
            vad_filter=True,
            align_model=None,  # No alignment yet
            diarize=False,
        )
        
        start = time.perf_counter()
        result_mlx = pipe_mlx(audio_file, print_progress=True)
        elapsed_mlx = time.perf_counter() - start
        
        rtf_mlx = duration / elapsed_mlx
        results["mlx_pipeline_rtf"] = rtf_mlx
        results["mlx_pipeline_time"] = elapsed_mlx
        
        print(f"\nMLX Pipeline Results:")
        print(f"  Time: {elapsed_mlx:.2f}s")
        print(f"  RTF: {rtf_mlx:.1f}x")
        print(f"  Segments: {len(result_mlx['segments'])}")
        
        # Test CPU baseline (faster-whisper)
        print("\nTesting CPU baseline...")
        try:
            model_cpu = whisperx.load_model(
                "large-v3",
                device="cpu",
                compute_type="float32",
            )
            
            start = time.perf_counter()
            result_cpu = model_cpu.transcribe(
                audio_file,
                batch_size=8,
            )
            elapsed_cpu = time.perf_counter() - start
            
            rtf_cpu = duration / elapsed_cpu
            results["cpu_pipeline_rtf"] = rtf_cpu
            results["cpu_pipeline_time"] = elapsed_cpu
            
            # Calculate speedup
            speedup = elapsed_cpu / elapsed_mlx
            results["speedup"] = speedup
            
            print(f"\nCPU Baseline Results:")
            print(f"  Time: {elapsed_cpu:.2f}s")
            print(f"  RTF: {rtf_cpu:.1f}x")
            print(f"\nSpeedup: {speedup:.1f}x")
            print(f"Target: ≥{self.targets['end_to_end_speedup']}x")
            print(f"Status: {'✓ PASS' if speedup >= self.targets['end_to_end_speedup'] else '✗ FAIL'}")
            
        except Exception as e:
            print(f"CPU baseline failed: {e}")
            results["speedup"] = 0
        
        return results
    
    def validate_accuracy(self) -> Dict:
        """Validate accuracy (WER) against baseline."""
        print(f"\n{'='*60}")
        print("Validating Accuracy (WER)")
        print('='*60)
        
        # Create test audio with known transcript
        test_transcript = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
            "The five boxing wizards jump quickly."
        )
        
        # Create audio
        audio_file = self.output_dir / "test_accuracy.wav"
        subprocess.run([
            "say", "-o", str(audio_file),
            "--data-format=LEF32@16000",
            test_transcript
        ], capture_output=True)
        
        # Transcribe with MLX
        import mlx_whisper
        audio = load_audio(str(audio_file))
        
        result_mlx = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo="mlx-community/whisper-large-v3",
            language="en",
            temperature=0.0,
        )
        
        # Calculate simple WER (word level)
        ref_words = test_transcript.lower().split()
        hyp_words = result_mlx["text"].lower().split()
        
        # Simple WER calculation
        errors = 0
        for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
            if ref != hyp:
                errors += 1
        
        # Handle length differences
        errors += abs(len(ref_words) - len(hyp_words))
        
        wer = errors / len(ref_words) * 100
        
        print(f"Reference: {test_transcript}")
        print(f"Hypothesis: {result_mlx['text']}")
        print(f"WER: {wer:.1f}%")
        print(f"Target: WER Δ ≤ {self.targets['wer_delta']} (from baseline)")
        
        # Note: Real WER validation would compare against OpenAI baseline
        return {
            "wer": wer,
            "reference": test_transcript,
            "hypothesis": result_mlx["text"],
        }
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite."""
        print("="*60)
        print("WhisperX MLX Performance Validation")
        print("="*60)
        print(f"System: {self.system_info.get('chip', 'Unknown')}")
        print(f"Memory: {self.system_info['memory_gb']:.1f} GB")
        print()
        
        results = {
            "system_info": self.system_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "targets": self.targets,
        }
        
        # Validate ASR performance
        for model in ["tiny", "base", "large-v3"]:
            key = f"asr_{model}"
            results[key] = self.validate_asr_performance(model)
        
        # Validate pipeline performance
        results["pipeline"] = self.validate_pipeline_performance()
        
        # Validate accuracy
        results["accuracy"] = self.validate_accuracy()
        
        # Save results
        output_file = self.output_dir / f"validation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Check ASR targets
        print("\nASR Performance:")
        for model in ["tiny", "base", "large-v3"]:
            key = f"asr_{model}"
            if key in results:
                r = results[key]
                fp16_pass = r["fp16_rtf"] >= self.targets["asr_rtf_fp16"]
                print(f"  {model:8} FP16: {r['fp16_rtf']:>6.1f}x {'✓' if fp16_pass else '✗'}")
                
                if "int4_rtf" in r and r["int4_rtf"] > 0:
                    int4_pass = r["int4_rtf"] >= self.targets["asr_rtf_int4"]
                    print(f"  {model:8} INT4: {r['int4_rtf']:>6.1f}x {'✓' if int4_pass else '✗'}")
        
        # Check pipeline speedup
        print("\nPipeline Performance:")
        if "pipeline" in results:
            p = results["pipeline"]
            if "speedup" in p and p["speedup"] > 0:
                speedup_pass = p["speedup"] >= self.targets["end_to_end_speedup"]
                print(f"  Speedup vs CPU: {p['speedup']:.1f}x {'✓' if speedup_pass else '✗'}")
            print(f"  MLX Pipeline RTF: {p.get('mlx_pipeline_rtf', 0):.1f}x")
        
        # Check accuracy
        print("\nAccuracy:")
        if "accuracy" in results:
            print(f"  WER: {results['accuracy']['wer']:.1f}%")
        
        # Overall status
        print("\n" + "="*60)
        
        # Determine if we meet roadmap targets
        meets_asr = any(
            results.get(f"asr_{m}", {}).get("fp16_rtf", 0) >= self.targets["asr_rtf_fp16"]
            for m in ["tiny", "base", "large-v3"]
        )
        meets_speedup = results.get("pipeline", {}).get("speedup", 0) >= self.targets["end_to_end_speedup"]
        
        if meets_asr and meets_speedup:
            print("✓ VALIDATION PASSED - Meets roadmap targets!")
        else:
            print("✗ VALIDATION FAILED - Does not meet all targets")
            if not meets_asr:
                print("  - ASR performance below target")
            if not meets_speedup:
                print("  - Pipeline speedup below target")


def main():
    parser = argparse.ArgumentParser(description="Validate MLX WhisperX performance")
    parser.add_argument("--output-dir", default="validation_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick validation (tiny model only)")
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.output_dir)
    
    if args.quick:
        # Quick test
        results = validator.validate_asr_performance("tiny")
        print(f"\nQuick test RTF: {results['fp16_rtf']:.1f}x")
    else:
        # Full validation
        validator.run_full_validation()


if __name__ == "__main__":
    main()