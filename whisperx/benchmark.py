import argparse
import os
import time
import torch
import torchaudio
import jiwer
import whisperx
import numpy as np
from typing import Tuple

def load_tedlium(root: str, download: bool = False, subset: str = "test"):
    print(f"Loading TEDLIUM dataset ({subset}) from {root}...")
    try:
        dataset = torchaudio.datasets.TEDLIUM(
            root=root,
            release="release3",
            subset=subset,
            download=download
        )
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def normalize_text(text: str) -> str:
    """
    Simple normalization: lower case, remove punctuation.
    """
    import string
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def benchmark(dataset, model_size="large-v2", device="cuda", compute_type="float16", batch_size=4, limit=None):
    print(f"Loading WhisperX model: {model_size} on {device} ({compute_type})...")
    
    try:
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model loaded.")
    
    total_wer = 0
    total_cer = 0
    total_latency = 0
    total_audio_duration = 0
    count = 0
    
    print(f"\nBenchmarking on {limit if limit else len(dataset)} samples...")

    # Clear CUDA cache for accurate VRAM measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial VRAM usage: {initial_vram:.2f} GB")

    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
            
        waveform, sample_rate, transcript, talk_id, speaker_id, identifier = item
        
        # WhisperX expects audio as a numpy array, float32, mono, 16kHz
        # TEDLIUM is likely 16kHz, but let's verify/resample if needed
        # waveform is (channels, time)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        audio_np = waveform.squeeze().numpy()
        
        duration = len(audio_np) / 16000
        total_audio_duration += duration

        # Measure Latency
        start_time = time.time()
        result = model.transcribe(audio_np, batch_size=batch_size)
        end_time = time.time()
        
        latency = end_time - start_time
        total_latency += latency
        
        # Combine segments for full transcript
        hypothesis = " ".join([seg['text'] for seg in result['segments']])
        
        # Normalize
        ref_norm = normalize_text(transcript)
        hyp_norm = normalize_text(hypothesis)
        
        if not ref_norm.strip():
            # Skip empty references to avoid division by zero in WER
            continue

        # Measure WER/CER
        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)
        
        total_wer += wer
        total_cer += cer
        count += 1
        
        print(f"Sample {i}: WER={wer:.2f}, CER={cer:.2f}, Latency={latency:.2f}s, Dur={duration:.2f}s, RTF={latency/duration:.2f}")

    if count == 0:
        print("No samples processed.")
        return

    avg_wer = total_wer / count
    avg_cer = total_cer / count
    avg_rtf = total_latency / total_audio_duration
    
    print("\n--- Benchmark Results ---")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average RTF (Real Time Factor): {avg_rtf:.4f}")
    print(f"Total Latency: {total_latency:.2f}s for {total_audio_duration:.2f}s audio")
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM Usage: {peak_vram:.2f} GB")
    else:
        print("VRAM Usage: N/A (CPU only)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark WhisperX on TEDLIUM")
    parser.add_argument("--root", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset if not found")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    args = parser.parse_args()
    
    # Create data dir
    os.makedirs(args.root, exist_ok=True)
    
    ds = load_tedlium(args.root, download=args.download)
    if ds:
        benchmark(ds, model_size=args.model, device=args.device, batch_size=args.batch_size, limit=args.limit)