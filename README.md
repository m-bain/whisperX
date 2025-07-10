<h1 align="center">WhisperX-MLX</h1>

<p align="center">
  <a href="https://github.com/sooth/whisperx-mlx/stargazers">
    <img src="https://img.shields.io/github/stars/sooth/whisperx-mlx.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/sooth/whisperx-mlx/issues">
        <img src="https://img.shields.io/github/issues/sooth/whisperx-mlx.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/sooth/whisperx-mlx/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/sooth/whisperx-mlx.svg"
             alt="GitHub license">
  </a>
  <a href="https://twitter.com/intent/tweet?text=&url=https%3A%2F%2Fgithub.com%2Fsooth%2Fwhisperx-mlx">
  <img src="https://img.shields.io/twitter/url/https/github.com/sooth/whisperx-mlx.svg?style=social" alt="Twitter">
  </a>      
</p>

<img width="1216" align="center" alt="whisperx-arch" src="https://raw.githubusercontent.com/m-bain/whisperX/refs/heads/main/figures/pipeline.png">

## WhisperX with MLX Backend for Apple Silicon

This is an enhanced fork of [WhisperX](https://github.com/m-bain/whisperX) that adds support for Apple's MLX framework, providing optimized performance on Apple Silicon Macs.

### Key Features

- 🚀 **MLX Backend Support**: Native Apple Silicon acceleration using the Metal Performance Shaders
- ⚡️ **Blazing Fast**: Up to 52x faster than real-time on M-series Macs
- 🎯 **Accurate Timestamps**: Word-level timestamp precision maintained from original WhisperX
- 📦 **Model Quantization**: INT4/INT8 quantization support for reduced memory usage
- 🔄 **Batch Processing**: 16% performance improvement with optimized batch processing
- 🎙️ **Full Compatibility**: Supports all WhisperX features including VAD and speaker diarization

### Performance Comparison

| Model | Backend | Precision | Batch | RTF | Speedup vs CPU |
|-------|---------|-----------|-------|-----|----------------|
| whisper-large-v3 | faster-whisper | INT8 | 8 | 1.2x | 1.0x |
| whisper-large-v3 | mlx | FP16 | 8 | 5.7x | **4.7x** |
| whisper-large-v3 | mlx | INT4 | 1 | 6.9x | **5.7x** |
| whisper-tiny | mlx | FP16 | 8 | 62.5x | **52.1x** |

### Installation

```bash
pip install whisperx-mlx
```

For development installation:
```bash
git clone https://github.com/sooth/whisperx-mlx.git
cd whisperx-mlx
pip install -e .
```

### Quick Start

```bash
# Basic usage with MLX backend
whisperx audio.mp3 --model large-v3 --backend mlx

# With INT4 quantization for reduced memory usage
whisperx audio.mp3 --model large-v3 --backend mlx --quantization int4

# Optimize for speed with batch processing
whisperx audio.mp3 --model large-v3 --backend mlx --batch_size 8
```

### Python API

```python
import whisperx

# Load model with MLX backend
model = whisperx.load_model("large-v3", backend="mlx", device="mps")

# Transcribe
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=8)

# Print results with word-level timestamps
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

### MLX-Specific Features

#### Model Quantization
```bash
# Use INT4 quantization (best speed/memory trade-off)
whisperx audio.mp3 --model large-v3 --backend mlx --quantization int4

# Use INT8 quantization (better accuracy)
whisperx audio.mp3 --model large-v3 --backend mlx --quantization int8
```

#### Process Separation
The MLX backend uses process separation to avoid conflicts between PyTorch (used for VAD) and MLX:
```python
from whisperx.process_separation import ProcessSeparatedPipeline

model = ProcessSeparatedPipeline(
    asr_backend="mlx",
    model_name="large-v3",
    batch_size=8,
    quantization="int4"
)
```

### Requirements

- macOS 13.0 or later  
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9-3.12

### Credits

This project builds upon:
- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) by the MLX team

### License

BSD 2-Clause License (same as original WhisperX)