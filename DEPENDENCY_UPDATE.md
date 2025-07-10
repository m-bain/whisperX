# Dependency Updates for WhisperX MLX Fork

## What Changed

### 1. **Custom mlx-whisper Fork**
We've created a fork of mlx-whisper with performance optimizations:
- Repository: https://github.com/sooth/mlx-whisper
- Branch: `whisperx-optimizations`
- Optimization: Avoids exact N_FRAMES boundaries for better quantized model performance

### 2. **Updated Dependencies**
The `pyproject.toml` now includes:
- `mlx>=0.26.0` - Apple's MLX framework
- `mlx-whisper @ git+https://github.com/sooth/mlx-whisper.git@whisperx-optimizations#subdirectory=whisper` - Our optimized fork
- `librosa>=0.10.0` - Audio processing library
- Removed `ctranslate2` and `faster-whisper` (not needed for MLX-only fork)

### 3. **Package Name Change**
- Changed from `whisperx` to `whisperx-mlx` to distinguish this as the MLX fork

## Installation from Fresh Clone

```bash
# Clone the repository
git clone https://github.com/sooth/whisperx-mlx.git
cd whisperx-mlx

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install with pip
pip install -e .
```

## Why These Changes?

1. **Performance**: The mlx-whisper fork includes optimizations for quantized models
2. **Clarity**: Clear separation between original WhisperX and MLX fork
3. **Dependencies**: Only includes what's needed for MLX operation

## Troubleshooting

If you encounter issues:

1. **Clear pip cache**: `pip cache purge`
2. **Reinstall**: `pip install --force-reinstall -e .`
3. **Check mlx-whisper**: Ensure our fork is being used: `pip show mlx-whisper`

The location should show it's installed from our GitHub fork.