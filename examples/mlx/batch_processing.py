#!/usr/bin/env python3
"""
Batch processing example for multiple audio files using MLX backend
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict
import whisperx
from whisperx.backends.mlx_whisper_v2 import MlxWhisperBackend


def process_batch(audio_files: List[str], model_size: str = "base", batch_size: int = 8):
    """
    Process multiple audio files in batch.
    
    Args:
        audio_files: List of audio file paths
        model_size: Model size
        batch_size: Batch size for processing
    """
    print(f"Processing {len(audio_files)} files with MLX backend")
    print(f"Model: {model_size}, Batch size: {batch_size}")
    
    # Initialize backend
    backend = MlxWhisperBackend(
        model=model_size,
        device="mlx",
        compute_type="float16",
        batch_size=batch_size,
        word_timestamps=True
    )
    
    results = []
    total_duration = 0
    total_time = 0
    
    # Process files
    for i, audio_file in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file}")
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_file)
            duration = len(audio) / 16000
            total_duration += duration
            
            # Transcribe
            start = time.time()
            result = backend.transcribe(
                audio,
                language="en",  # Set to None for auto-detection
                print_progress=False
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            # Store result
            results.append({
                "file": audio_file,
                "duration": duration,
                "time": elapsed,
                "rtf": duration / elapsed,
                "result": result
            })
            
            print(f"  Duration: {duration:.1f}s")
            print(f"  Processing time: {elapsed:.2f}s")
            print(f"  RTF: {duration/elapsed:.1f}x")
            print(f"  Text preview: {result['text'][:100]}...")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "file": audio_file,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Files processed: {len([r for r in results if 'error' not in r])}/{len(audio_files)}")
    print(f"Total audio duration: {total_duration:.1f}s")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average RTF: {total_duration/total_time:.1f}x")
    
    return results


def process_directory(directory: str, pattern: str = "*.mp3", **kwargs):
    """
    Process all audio files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        **kwargs: Arguments for process_batch
    """
    audio_files = list(Path(directory).glob(pattern))
    
    if not audio_files:
        print(f"No files matching {pattern} found in {directory}")
        return
    
    print(f"Found {len(audio_files)} files matching {pattern}")
    
    # Process in batches
    results = process_batch([str(f) for f in audio_files], **kwargs)
    
    # Save results
    output_dir = Path(directory) / "transcriptions"
    output_dir.mkdir(exist_ok=True)
    
    for result in results:
        if "error" in result:
            continue
            
        # Save transcription
        base_name = Path(result["file"]).stem
        output_file = output_dir / f"{base_name}_transcript.txt"
        
        with open(output_file, "w") as f:
            f.write(f"File: {result['file']}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            f.write(f"Processing time: {result['time']:.2f}s\n")
            f.write(f"RTF: {result['rtf']:.1f}x\n")
            f.write(f"\nTranscription:\n")
            f.write(result["result"]["text"])
        
        print(f"Saved: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_processing.py <audio_file1> <audio_file2> ...")
        print("  python batch_processing.py --dir <directory> [--pattern '*.wav']")
        print("\nOptions:")
        print("  --model MODEL     Model size (default: base)")
        print("  --batch-size N    Batch size (default: 8)")
        print("  --pattern PAT     File pattern for directory mode (default: *.mp3)")
        sys.exit(1)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Audio files to process")
    parser.add_argument("--dir", help="Process all files in directory")
    parser.add_argument("--pattern", default="*.mp3", help="File pattern for directory mode")
    parser.add_argument("--model", default="base", help="Model size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    if args.dir:
        # Directory mode
        process_directory(
            args.dir,
            pattern=args.pattern,
            model_size=args.model,
            batch_size=args.batch_size
        )
    elif args.files:
        # File list mode
        results = process_batch(
            args.files,
            model_size=args.model,
            batch_size=args.batch_size
        )
        
        # Save results
        for result in results:
            if "error" not in result:
                output = result["file"].rsplit(".", 1)[0] + "_transcript.txt"
                with open(output, "w") as f:
                    f.write(result["result"]["text"])
                print(f"Saved: {output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()