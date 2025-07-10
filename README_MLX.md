# WhisperX MLX Fork

Fast automatic speech recognition with word-level timestamps, optimized for Apple Silicon using MLX.

This is an MLX-only fork of [WhisperX](https://github.com/m-bain/whisperX) that runs exclusively on Apple Silicon Macs using the MLX framework.

## Features

- 🚀 **Apple Silicon Optimized** - Runs natively on M1/M2/M3 chips using MLX
- ⚡ **Batch Processing** - ~16% faster transcription with parallel processing
- 🎯 **Word-Level Timestamps** - Accurate word-level timing information
- 🗣️ **VAD Integration** - Voice Activity Detection with Silero or PyAnnote
- 📦 **INT4 Quantization** - Reduced memory usage and faster inference
- 🌍 **Multilingual** - Supports 100+ languages

## Quick Install

```bash
# Clone the repository
git clone https://github.com/sooth/whisperx-mlx.git
cd whisperx-mlx

# Install
pip install -e .
```

## Quick Start

### Command Line
```bash
# Basic transcription
whisperx audio.mp3 --model large-v3

# Fast batch processing
whisperx audio.mp3 --model large-v3 --backend batch

# INT4 quantized (faster, less memory)
whisperx audio.mp3 --model large-v3 --compute_type int4
```

### Python
```python
import whisperx

# Load model
model = whisperx.load_model("large-v3", backend="mlx")

# Transcribe
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=8)

# Print results
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s] {segment['text']}")
```

## Why This Fork?

- **MLX-Only**: Simplified codebase focused solely on Apple Silicon
- **Performance**: Optimized batch processing for faster transcription
- **Compatibility**: Drop-in replacement for WhisperX on macOS
- **Maintained**: Active development for MLX backend improvements

## Documentation

- [Getting Started Guide](GETTING_STARTED.md) - Detailed setup and usage
- [MLX Fork Status](MLX_FORK_STATUS.md) - Technical details and features
- [Test Suite](test_mlx_complete.py) - Verify your installation

## Models

All standard Whisper models are supported:
- `tiny`, `base`, `small`, `medium`
- `large`, `large-v2`, `large-v3`
- INT4 quantized versions for faster inference

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- macOS 12.0 or later
- Python 3.8+

## License

This fork maintains the original WhisperX BSD 4-Clause License.

## Acknowledgements

- Original [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) implementation