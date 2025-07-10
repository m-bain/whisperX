# Getting Started with WhisperX MLX Fork

This guide will help you get up and running with the WhisperX MLX fork from a fresh clone.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- Python 3.8 or higher
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sooth/whisperx-mlx.git
cd whisperx-mlx
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install WhisperX

```bash
pip install -e .
```

This will install WhisperX and all required dependencies including:
- mlx-whisper (Apple Silicon optimized Whisper)
- PyTorch (for VAD models)
- NumPy, librosa, and other audio processing libraries

### 4. Download Models (Optional)

Models will be automatically downloaded on first use, but you can pre-download them:

```python
import whisperx
# This will download the model
model = whisperx.load_model("large-v3", backend="mlx")
```

## Quick Start

### Command Line Usage

Basic transcription:
```bash
whisperx audio.mp3 --model large-v3 --backend mlx
```

With output format:
```bash
whisperx audio.mp3 --model large-v3 --backend mlx --output_format srt
```

Using batch processing for better performance:
```bash
whisperx audio.mp3 --model large-v3 --backend batch --batch_size 16
```

Using INT4 quantized model (faster, less memory):
```bash
whisperx audio.mp3 --model large-v3 --compute_type int4
```

### Python API Usage

```python
import whisperx

# Load model
model = whisperx.load_model("large-v3", device="cpu", backend="mlx")

# Load audio
audio = whisperx.load_audio("audio.mp3")

# Transcribe
result = model.transcribe(audio, batch_size=8)

# Print results
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

## Available Models

- `tiny` - Fastest, least accurate
- `base` - Good balance
- `small` - Better accuracy
- `medium` - Even better accuracy
- `large` - Best accuracy (older)
- `large-v2` - Improved large model
- `large-v3` - Latest and best model

## Backend Options

- `mlx` or `standard` - Standard MLX backend
- `batch` - Optimized batch processing (~16% faster)
- `auto` - Automatically selects best backend

## Common Options

```bash
# Specify language (auto-detect by default)
whisperx audio.mp3 --language en

# Disable text alignment (faster)
whisperx audio.mp3 --no_align

# Adjust VAD sensitivity
whisperx audio.mp3 --vad_onset 0.4 --vad_offset 0.3

# Use different VAD method
whisperx audio.mp3 --vad_method silero  # or pyannote

# Multiple output formats
whisperx audio.mp3 --output_format all  # Creates srt, vtt, txt, json, etc.
```

## Troubleshooting

### 1. Import Errors

If you get import errors, ensure you installed in editable mode:
```bash
pip install -e .
```

### 2. Model Download Issues

If model download fails, you can manually specify a local path:
```python
model = whisperx.load_model("path/to/local/model", backend="mlx")
```

### 3. Memory Issues

Use a smaller model or INT4 quantization:
```bash
whisperx audio.mp3 --model tiny --compute_type int4
```

### 4. VAD Issues

Try different VAD settings:
```bash
# More aggressive VAD (catches more speech)
whisperx audio.mp3 --vad_onset 0.3 --vad_offset 0.2

# Less aggressive VAD
whisperx audio.mp3 --vad_onset 0.6 --vad_offset 0.4

# Switch VAD method
whisperx audio.mp3 --vad_method silero
```

## Testing Your Installation

Run the included test script:
```bash
python test_mlx_complete.py
```

Or test fresh clone functionality:
```bash
./test_fresh_clone.sh
```

## Examples

### Transcribe with Timestamps
```python
import whisperx

model = whisperx.load_model("base", backend="mlx")
audio = whisperx.load_audio("interview.mp3")
result = model.transcribe(audio)

# Save with timestamps
with open("transcript.txt", "w") as f:
    for segment in result["segments"]:
        f.write(f"[{segment['start']:.2f}s] {segment['text']}\n")
```

### Batch Process Multiple Files
```python
import whisperx
import glob

model = whisperx.load_model("small", backend="batch")

for audio_file in glob.glob("*.mp3"):
    print(f"Processing {audio_file}...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    
    # Save as SRT
    output_file = audio_file.replace(".mp3", ".srt")
    # ... write SRT format
```

### Use INT4 Quantization
```bash
# For faster inference with slightly reduced accuracy
whisperx podcast.mp3 --model large-v3 --compute_type int4 --backend batch
```

## Performance Tips

1. **Use Batch Backend**: Add `--backend batch` for ~16% speed improvement
2. **Use INT4 Models**: Add `--compute_type int4` for faster inference
3. **Disable Alignment**: Add `--no_align` if you don't need word-level timestamps
4. **Adjust Batch Size**: Larger batch sizes (8-16) are generally faster
5. **Choose Right Model**: Use the smallest model that meets your accuracy needs

## Next Steps

- Check out the [MLX_FORK_STATUS.md](MLX_FORK_STATUS.md) for technical details
- Read the original [WhisperX documentation](https://github.com/m-bain/whisperX)
- Explore the [mlx-whisper documentation](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

## Support

For issues specific to this MLX fork, please open an issue on the [GitHub repository](https://github.com/sooth/whisperx-mlx).