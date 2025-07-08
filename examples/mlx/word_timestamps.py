#!/usr/bin/env python3
"""
Word-level timestamp extraction using MLX backend
"""

import sys
import json
from typing import List, Dict
import whisperx


def extract_word_timestamps(audio_file: str, model_size: str = "base"):
    """
    Extract word-level timestamps from audio.
    
    Args:
        audio_file: Path to audio file
        model_size: Model size
    """
    print(f"Extracting word timestamps from: {audio_file}")
    
    # Load model with word timestamps enabled
    model = whisperx.load_model(
        model_size,
        device="mlx",
        compute_type="float16",
        asr_backend="mlx"
    )
    
    # Load audio
    audio = whisperx.load_audio(audio_file)
    duration = len(audio) / 16000
    
    # Transcribe with word timestamps
    print("Transcribing with word timestamps...")
    result = model.transcribe(
        audio,
        word_timestamps=True,
        language="en"
    )
    
    # Extract words
    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            all_words.extend(segment["words"])
    
    print(f"\nFound {len(all_words)} words in {duration:.1f}s audio")
    
    return result, all_words


def create_subtitle_file(words: List[Dict], output_file: str, max_words: int = 10):
    """
    Create SRT subtitle file from word timestamps.
    
    Args:
        words: List of word dictionaries with timestamps
        output_file: Output SRT file path
        max_words: Maximum words per subtitle line
    """
    with open(output_file, "w", encoding="utf-8") as f:
        subtitle_idx = 1
        
        # Group words into subtitle lines
        for i in range(0, len(words), max_words):
            word_group = words[i:i + max_words]
            
            if not word_group:
                continue
            
            # Get timing
            start_time = word_group[0]["start"]
            end_time = word_group[-1]["end"]
            
            # Format time
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
            
            # Write subtitle
            f.write(f"{subtitle_idx}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(" ".join(w["word"] for w in word_group) + "\n\n")
            
            subtitle_idx += 1
    
    print(f"Created subtitle file: {output_file}")


def create_json_output(result: Dict, words: List[Dict], output_file: str):
    """
    Create detailed JSON output with all timestamps.
    
    Args:
        result: Full transcription result
        words: List of words with timestamps
        output_file: Output JSON file path
    """
    output = {
        "text": result["text"],
        "language": result.get("language", "unknown"),
        "duration": words[-1]["end"] if words else 0,
        "word_count": len(words),
        "segments": result["segments"],
        "words": words
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Created JSON output: {output_file}")


def analyze_speech_rate(words: List[Dict], window_size: int = 30):
    """
    Analyze speech rate over time.
    
    Args:
        words: List of words with timestamps
        window_size: Window size in seconds
    """
    if not words:
        return
    
    print("\nSpeech Rate Analysis:")
    print("=" * 40)
    
    # Calculate words per minute in windows
    current_window_start = 0
    window_words = []
    
    for word in words:
        if word["start"] >= current_window_start + window_size:
            # Calculate rate for current window
            if window_words:
                wpm = len(window_words) * 60 / window_size
                print(f"{current_window_start:3.0f}s - {current_window_start + window_size:3.0f}s: "
                      f"{wpm:3.0f} words/min ({len(window_words)} words)")
            
            # Move to next window
            current_window_start += window_size
            window_words = []
        
        window_words.append(word)
    
    # Last window
    if window_words:
        actual_duration = words[-1]["end"] - current_window_start
        if actual_duration > 0:
            wpm = len(window_words) * 60 / actual_duration
            print(f"{current_window_start:3.0f}s - {words[-1]['end']:3.0f}s: "
                  f"{wpm:3.0f} words/min ({len(window_words)} words)")
    
    # Overall statistics
    total_duration = words[-1]["end"] - words[0]["start"]
    overall_wpm = len(words) * 60 / total_duration
    print(f"\nOverall: {overall_wpm:.0f} words/min")


def main():
    if len(sys.argv) < 2:
        print("Usage: python word_timestamps.py <audio_file> [model_size]")
        print("\nExample: python word_timestamps.py podcast.mp3 base")
        print("\nThis will create:")
        print("  - podcast_words.json (detailed word timestamps)")
        print("  - podcast_words.srt (subtitle file)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
    
    try:
        # Extract word timestamps
        result, words = extract_word_timestamps(audio_file, model_size)
        
        if not words:
            print("No word timestamps found. Make sure the model supports word timestamps.")
            sys.exit(1)
        
        # Create output files
        base_name = audio_file.rsplit(".", 1)[0]
        
        # Save JSON
        json_file = f"{base_name}_words.json"
        create_json_output(result, words, json_file)
        
        # Save SRT
        srt_file = f"{base_name}_words.srt"
        create_subtitle_file(words, srt_file)
        
        # Analyze speech rate
        analyze_speech_rate(words)
        
        # Print sample
        print("\nSample word timestamps:")
        print("=" * 60)
        for word in words[:10]:
            print(f"{word['start']:6.2f}s - {word['end']:6.2f}s: {word['word']}")
        if len(words) > 10:
            print(f"... and {len(words) - 10} more words")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()