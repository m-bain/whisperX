#!/usr/bin/env python3
"""
Model conversion script for WhisperX MLX backend
Uses official mlx-examples conversion tools
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def convert_whisper_model(
    torch_name_or_path: str,
    mlx_path: str,
    dtype: str = "float16",
    quantize: bool = False,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> bool:
    """Convert PyTorch Whisper model to MLX format.
    
    Uses the official mlx-examples/whisper/convert.py script.
    
    Args:
        torch_name_or_path: HuggingFace model ID or local path
        mlx_path: Output path for MLX model
        dtype: Model precision ("float16" or "float32")
        quantize: Whether to quantize to INT4
        q_bits: Quantization bits (4 or 8)
        q_group_size: Quantization group size
        
    Returns:
        True if successful, False otherwise
    """
    # Check if mlx-examples is available
    mlx_examples_path = Path("mlx-examples")
    if not mlx_examples_path.exists():
        print("Cloning mlx-examples...")
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/ml-explore/mlx-examples.git"
            ], check=True)
        except subprocess.CalledProcessError:
            print("Failed to clone mlx-examples")
            return False
    
    # Path to conversion script
    convert_script = mlx_examples_path / "whisper" / "convert.py"
    
    if not convert_script.exists():
        print(f"Conversion script not found at {convert_script}")
        return False
    
    # Build conversion command
    cmd = [
        sys.executable,
        str(convert_script),
        "--torch-name-or-path", torch_name_or_path,
        "--mlx-path", mlx_path,
        "--dtype", dtype,
    ]
    
    if quantize:
        cmd.extend(["-q", "--q-bits", str(q_bits), "--q-group-size", str(q_group_size)])
    
    print(f"Converting {torch_name_or_path} to MLX format...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Model converted successfully to {mlx_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        return False


def benchmark_model(mlx_path: str, audio_file: Optional[str] = None) -> float:
    """Benchmark converted MLX model.
    
    Args:
        mlx_path: Path to MLX model
        audio_file: Audio file for benchmarking
        
    Returns:
        Real-time factor (RTF)
    """
    print(f"\nBenchmarking {mlx_path}...")
    
    # Create test audio if not provided
    if audio_file is None:
        audio_file = "test_benchmark.wav"
        if not Path(audio_file).exists():
            print("Creating test audio...")
            subprocess.run([
                "say", "-o", audio_file,
                "--data-format=LEF32@16000",
                "This is a test of the WhisperX MLX backend. " * 10
            ], capture_output=True)
    
    # Benchmark script
    benchmark_code = f"""
import mlx_whisper
import soundfile as sf
import time
import numpy as np

# Load audio
audio, sr = sf.read("{audio_file}")
duration = len(audio) / sr

# Load model
model = mlx_whisper.load_model("{mlx_path}")

# Warm-up
_ = mlx_whisper.transcribe(audio[:sr], model=model)

# Benchmark
start = time.perf_counter()
result = mlx_whisper.transcribe(audio, model=model, verbose=False)
elapsed = time.perf_counter() - start

rtf = duration / elapsed
print(f"Duration: {{duration:.1f}}s")
print(f"Elapsed: {{elapsed:.2f}}s")
print(f"RTF: {{rtf:.1f}}x")
print(f"RTF_VALUE={{rtf}}")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", benchmark_code],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract RTF from output
        for line in result.stdout.split('\n'):
            if line.startswith("RTF_VALUE="):
                rtf = float(line.split('=')[1])
                return rtf
        
        print(result.stdout)
        return 0.0
        
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        print(f"Error: {e.stderr}")
        return 0.0


def convert_all_models(output_dir: str = "~/mlx_models"):
    """Convert all standard Whisper models to MLX format."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    models = [
        ("openai/whisper-tiny", "tiny", False),
        ("openai/whisper-tiny.en", "tiny.en", False),
        ("openai/whisper-base", "base", False),
        ("openai/whisper-base.en", "base.en", False),
        ("openai/whisper-small", "small", False),
        ("openai/whisper-small.en", "small.en", False),
        ("openai/whisper-medium", "medium", False),
        ("openai/whisper-medium.en", "medium.en", False),
        ("openai/whisper-large-v3", "large-v3", False),
        # INT4 versions
        ("openai/whisper-tiny", "tiny-int4", True),
        ("openai/whisper-base", "base-int4", True),
        ("openai/whisper-small", "small-int4", True),
        ("openai/whisper-large-v3", "large-v3-int4", True),
    ]
    
    results = []
    
    for hf_model, name, quantize in models:
        mlx_path = os.path.join(output_dir, name)
        
        if Path(mlx_path).exists():
            print(f"\n{name} already exists, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {name}...")
        print('='*60)
        
        success = convert_whisper_model(
            hf_model,
            mlx_path,
            dtype="float16",
            quantize=quantize,
            q_bits=4,
            q_group_size=64
        )
        
        if success:
            rtf = benchmark_model(mlx_path)
            results.append((name, rtf, "✓"))
        else:
            results.append((name, 0.0, "✗"))
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    
    for name, rtf, status in results:
        if rtf > 0:
            print(f"{status} {name:20} RTF: {rtf:>6.1f}x")
        else:
            print(f"{status} {name:20} Failed")
    
    print(f"\nModels saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Whisper models to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single model
  python convert_models.py --torch-name openai/whisper-large-v3 --mlx-path ~/mlx_models/large-v3

  # Convert with INT4 quantization
  python convert_models.py --torch-name openai/whisper-large-v3 --mlx-path ~/mlx_models/large-v3-int4 -q

  # Convert all standard models
  python convert_models.py --all --output-dir ~/mlx_models

  # Benchmark existing model
  python convert_models.py --benchmark ~/mlx_models/large-v3
""")
    
    parser.add_argument("--torch-name", "--torch-name-or-path", 
                       help="HuggingFace model ID or local path")
    parser.add_argument("--mlx-path", 
                       help="Output path for MLX model")
    parser.add_argument("--dtype", default="float16", 
                       choices=["float16", "float32"],
                       help="Model precision")
    parser.add_argument("-q", "--quantize", action="store_true",
                       help="Quantize model to INT4")
    parser.add_argument("--q-bits", type=int, default=4,
                       help="Quantization bits")
    parser.add_argument("--q-group-size", type=int, default=64,
                       help="Quantization group size")
    parser.add_argument("--all", action="store_true",
                       help="Convert all standard models")
    parser.add_argument("--output-dir", default="~/mlx_models",
                       help="Output directory for --all")
    parser.add_argument("--benchmark", 
                       help="Benchmark existing MLX model")
    parser.add_argument("--audio", 
                       help="Audio file for benchmarking")
    
    args = parser.parse_args()
    
    if args.benchmark:
        rtf = benchmark_model(args.benchmark, args.audio)
        if rtf > 0:
            print(f"\nBenchmark result: {rtf:.1f}x real-time")
    elif args.all:
        convert_all_models(args.output_dir)
    elif args.torch_name and args.mlx_path:
        success = convert_whisper_model(
            args.torch_name,
            args.mlx_path,
            args.dtype,
            args.quantize,
            args.q_bits,
            args.q_group_size
        )
        if success and not args.benchmark:
            rtf = benchmark_model(args.mlx_path, args.audio)
            if rtf > 0:
                print(f"\nBenchmark result: {rtf:.1f}x real-time")
    else:
        parser.error("Specify --torch-name and --mlx-path, or use --all, or --benchmark")


if __name__ == "__main__":
    main()