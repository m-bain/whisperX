#!/usr/bin/env python3
"""
Basic transcription example using WhisperX MLX backend
"""

import sys
import time
import whisperx

def transcribe_with_mlx(audio_file: str, model_size: str = "base"):
    """
    Transcribe audio using MLX backend.
    
    Args:
        audio_file: Path to audio file
        model_size: Model size (tiny, base, small, large-v3)
    """
    print(f"Transcribing {audio_file} with MLX backend...")
    print(f"Model: {model_size}")
    
    # Load model
    start = time.time()
    device = "mlx"
    compute_type = "float16"
    
    print(f"\nLoading model...")
    model = whisperx.load_model(
        model_size, 
        device=device,
        compute_type=compute_type,
        asr_backend="mlx"
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Load audio
    print(f"\nLoading audio...")
    audio = whisperx.load_audio(audio_file)
    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.1f}s")
    
    # Transcribe
    print(f"\nTranscribing...")
    start = time.time()
    result = model.transcribe(
        audio,
        batch_size=8,
        language="en",  # Set to None for auto-detection
        print_progress=True
    )
    transcribe_time = time.time() - start
    
    # Calculate metrics
    rtf = duration / transcribe_time
    print(f"\nTranscription completed in {transcribe_time:.2f}s")
    print(f"Real-time factor: {rtf:.1f}x")
    
    # Print results
    print("\n" + "="*60)
    print("TRANSCRIPTION")
    print("="*60)
    
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
    
    print("\n" + "="*60)
    print("FULL TEXT")
    print("="*60)
    print(result["text"])
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Detected language: {result.get('language', 'unknown')}")
    print(f"Number of segments: {len(result['segments'])}")
    print(f"Total words: {sum(len(seg['text'].split()) for seg in result['segments'])}")
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python basic_transcription.py <audio_file> [model_size]")
        print("Model sizes: tiny, base, small, large-v3")
        print("Example: python basic_transcription.py audio.mp3 base")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    # For INT4 models, append -q4
    if model_size.endswith("-q4"):
        print("Using INT4 quantized model for faster inference")
    
    try:
        result = transcribe_with_mlx(audio_file, model_size)
        
        # Save output
        output_file = audio_file.rsplit(".", 1)[0] + "_mlx_transcription.txt"
        with open(output_file, "w") as f:
            f.write(result["text"])
        print(f"\nTranscription saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()