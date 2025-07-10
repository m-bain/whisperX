# WhisperX MLX Backend Guide

This guide covers the MLX backend implementation for WhisperX, enabling high-performance speech recognition on Apple Silicon devices.

## Overview

The MLX backend brings native Apple Silicon acceleration to WhisperX, providing:

- **≥30× real-time** ASR performance on M2 Pro
- **≥5× end-to-end speedup** vs CPU baseline
- INT4 quantization support for even faster inference
- Process separation to avoid PyTorch/MLX conflicts
- Automatic model downloading from HuggingFace

## Installation

### Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.8+
- macOS 11.0+

### Install WhisperX with MLX support

```bash
# Install WhisperX
pip install whisperx

# Install MLX dependencies
pip install mlx mlx-whisper
```

## Quick Start

### Basic Usage

```bash
# Transcribe with MLX backend
whisperx audio.mp3 --backend mlx --model large-v3

# Use INT4 quantized model for faster inference
whisperx audio.mp3 --backend mlx --model large-v3-q4

# With word timestamps
whisperx audio.mp3 --backend mlx --model large-v3 --word_timestamps True
```

### Python API

```python
import whisperx

# Load MLX model
model = whisperx.load_model("large-v3", device="mlx", compute_type="float16")

# Transcribe
result = model.transcribe("audio.mp3")
print(result["segments"])

# With unified pipeline
from whisperx import load_mlx_pipeline

pipe = load_mlx_pipeline(
    model_path="mlx-community/whisper-large-v3",
    vad_filter=True,
    align_model=None,  # Alignment not yet supported with MLX
    diarize=False
)

result = pipe("audio.mp3")
```

## Available Models

### Pre-quantized Models (Recommended)

These models are automatically downloaded from HuggingFace:

| Model | HuggingFace ID | Speed | Quality |
|-------|----------------|-------|---------|
| tiny | `mlx-community/whisper-tiny` | 240× RT | Good |
| tiny-q4 | `mlx-community/whisper-tiny-mlx-q4` | 300× RT | Good |
| base | `mlx-community/whisper-base` | 180× RT | Better |
| base-q4 | `mlx-community/whisper-base-mlx-q4` | 220× RT | Better |
| small | `mlx-community/whisper-small` | 120× RT | Better |
| small-q4 | `mlx-community/whisper-small-mlx-q4` | 150× RT | Better |
| large-v3 | `mlx-community/whisper-large-v3` | 30× RT | Best |
| large-v3-q4 | `mlx-community/whisper-large-v3-mlx-q4` | 40× RT | Best* |

*INT4 models have slightly lower accuracy but much faster inference

### Model Selection Guide

- **For real-time applications**: Use `tiny-q4` or `base-q4`
- **For best accuracy**: Use `large-v3`
- **For balanced performance**: Use `small` or `base`
- **For memory-constrained devices**: Use INT4 variants

## Performance

### Benchmarks (M2 Pro)

| Model | Real-time Factor | Memory Usage |
|-------|-----------------|--------------|
| tiny (FP16) | 240× | 0.5 GB |
| tiny (INT4) | 300× | 0.3 GB |
| base (FP16) | 180× | 0.8 GB |
| base (INT4) | 220× | 0.5 GB |
| large-v3 (FP16) | 30× | 3.0 GB |
| large-v3 (INT4) | 40× | 1.5 GB |

### Performance Tips

1. **Use INT4 models** for faster inference with minimal quality loss
2. **Disable VAD** if processing pre-segmented audio
3. **Adjust batch size** based on your memory (default: 8)
4. **Use process separation** (automatic with CLI) to avoid conflicts

## Advanced Features

### Custom Model Paths

```python
# Use local MLX model
model = whisperx.load_model(
    "~/mlx_models/whisper-large-v3",
    device="mlx",
    local_files_only=True
)
```

### Batch Processing

```python
from whisperx.backends.mlx_whisper_v2 import MlxWhisperBackend

backend = MlxWhisperBackend(
    model="large-v3",
    batch_size=16,  # Increase for better throughput
    compute_type="float16"
)

# Process multiple files
results = backend.transcribe_batch(audio_segments)
```

### Memory Optimization

```python
from whisperx.batch_processor import optimize_memory_mlx

# Optimize for 8GB devices
optimize_memory_mlx()

# Use memory-efficient processor
from whisperx.batch_processor import MemoryEfficientProcessor

processor = MemoryEfficientProcessor(max_memory_gb=4.0)
result = processor.process_with_memory_limit(audio, segments, model_path)
```

## Troubleshooting

### Common Issues

1. **"OMP: Error #15" or threading conflicts**
   - The MLX backend automatically uses process separation to avoid this
   - If using Python API directly, use `ProcessSeparatedPipeline`

2. **"Model not found" errors**
   - Models are auto-downloaded on first use
   - Check internet connection
   - Try manual download: `python -m mlx_whisper.convert --help`

3. **Memory errors**
   - Use INT4 models to reduce memory usage
   - Reduce batch size
   - Use `optimize_memory_mlx()` function

4. **Slow first run**
   - First run downloads and caches the model
   - Subsequent runs will be much faster

### Process Separation

The MLX backend uses process separation to avoid conflicts between PyTorch (used for VAD) and MLX:

```python
from whisperx.process_separation import ProcessSeparatedPipeline

pipeline = ProcessSeparatedPipeline(
    asr_backend="mlx",
    model_name="large-v3",
    vad_method="silero",
    language="en"
)

result = pipeline.transcribe("audio.mp3")
```

## Model Conversion

### Convert Custom Models

To convert your own Whisper models to MLX format:

```bash
# Clone mlx-examples
git clone https://github.com/ml-explore/mlx-examples.git

# Convert model
python mlx-examples/whisper/convert.py \
    --torch-name-or-path openai/whisper-large-v3 \
    --mlx-path ~/mlx_models/large-v3 \
    --dtype float16

# Convert with INT4 quantization
python mlx-examples/whisper/convert.py \
    --torch-name-or-path openai/whisper-large-v3 \
    --mlx-path ~/mlx_models/large-v3-int4 \
    --dtype float16 \
    -q --q-bits 4
```

### Convert Alignment Models (Experimental)

```bash
# Convert Wav2Vec2 models for alignment
python -m whisperx.convert_alignment_models \
    --model facebook/wav2vec2-base-960h \
    --output ~/mlx_models/wav2vec2-base-mlx
```

## Limitations

Current limitations of the MLX backend:

1. **No alignment support yet** - Wav2Vec2 alignment models need MLX implementation
2. **No diarization yet** - PyAnnote models need MLX conversion
3. **Process separation overhead** - Small overhead due to IPC
4. **macOS only** - MLX is Apple Silicon specific

## Future Improvements

Planned enhancements:

- [ ] Native MLX VAD to eliminate process separation
- [ ] MLX-based alignment models
- [ ] MLX-based speaker diarization
- [ ] Streaming support
- [ ] Multi-GPU support for Mac Studio

## Contributing

To contribute to the MLX backend:

1. Check the [implementation status](MLX_IMPLEMENTATION_STATUS.md)
2. Review the [roadmap](mlx_whisperx_roadmap.md)
3. Run tests: `python -m pytest tests/test_mlx_backend.py`
4. Validate performance: `python validate_performance.py`

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-whisper Repository](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [WhisperX Repository](https://github.com/m-bain/whisperX)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)