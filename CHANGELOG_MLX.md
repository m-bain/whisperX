# WhisperX-MLX Changelog

## Major Enhancements over Original WhisperX

### MLX Backend Implementation
- Added complete MLX backend support for Apple Silicon acceleration
- Achieves up to 52x real-time performance on M-series Macs
- Maintains full compatibility with existing WhisperX features

### Model Quantization Support
- INT4 quantization: 19.6x faster than CPU baseline with minimal accuracy loss
- INT8 quantization: Better accuracy with good performance
- Dynamic quantization loading with `--quantization` flag

### Batch Processing Optimization
- Implemented true batch processing with 16% performance improvement
- Smart segment grouping by length to minimize padding overhead
- Optimal batch size of 8 for best performance

### Process Separation Architecture
- Separated VAD (PyTorch) and ASR (MLX) processes to avoid framework conflicts
- Enables stable operation with both PyTorch and MLX in the same pipeline
- Automatic process management with proper cleanup

### Enhanced CLI Support
- `--backend mlx`: Enable MLX backend
- `--quantization {int4,int8}`: Use quantized models
- `--batch_size N`: Configure batch processing
- Full compatibility with existing WhisperX CLI options

### Performance Optimizations
- Fixed chunk boundary performance bug in MLX
- Memory-efficient model loading
- Optimized audio preprocessing for Metal Performance Shaders

### Bug Fixes
- Fixed PyTorch 2.6+ compatibility for VAD models
- Resolved word timestamp accuracy issues
- Fixed beam search compatibility with MLX

## Installation

```bash
pip install whisperx-mlx
```

## Migration from WhisperX

Simply add `--backend mlx` to your existing WhisperX commands:

```bash
# Original WhisperX
whisperx audio.mp3 --model large-v3

# WhisperX-MLX
whisperx audio.mp3 --model large-v3 --backend mlx
```

All other options and features work exactly the same!