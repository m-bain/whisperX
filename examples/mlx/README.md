# WhisperX MLX Examples

This directory contains example scripts demonstrating how to use the MLX backend for WhisperX on Apple Silicon devices.

## Examples

### 1. Basic Transcription (`basic_transcription.py`)

Simple transcription of audio files using the MLX backend.

```bash
# Basic usage
python basic_transcription.py audio.mp3

# With specific model
python basic_transcription.py audio.mp3 large-v3

# With INT4 quantized model
python basic_transcription.py audio.mp3 tiny-q4
```

### 2. Batch Processing (`batch_processing.py`)

Process multiple audio files efficiently.

```bash
# Process multiple files
python batch_processing.py file1.mp3 file2.wav file3.m4a

# Process entire directory
python batch_processing.py --dir /path/to/audio --pattern "*.mp3"

# With custom settings
python batch_processing.py --dir /path/to/audio --model large-v3 --batch-size 16
```

### 3. Word Timestamps (`word_timestamps.py`)

Extract word-level timestamps and create subtitles.

```bash
# Extract word timestamps
python word_timestamps.py podcast.mp3

# With specific model
python word_timestamps.py interview.wav large-v3
```

This creates:
- `*.json` - Detailed word timestamp data
- `*.srt` - Subtitle file

### 4. Benchmark (`benchmark.py`)

Compare performance of different models.

```bash
# Compare default models
python benchmark.py test_audio.mp3

# Compare specific models
python benchmark.py test_audio.mp3 tiny base large-v3

# Compare FP16 vs INT4
python benchmark.py test_audio.mp3 base base-q4 large-v3 large-v3-q4
```

## Performance Tips

1. **Model Selection**:
   - Use `tiny` or `tiny-q4` for real-time applications
   - Use `base` or `small` for balanced performance
   - Use `large-v3` for best accuracy
   - Use INT4 (`-q4`) models for faster inference

2. **Memory Usage**:
   - INT4 models use ~50% less memory
   - Reduce batch size if running out of memory
   - Close other applications when using large models

3. **Speed Optimization**:
   - Disable VAD if audio is pre-segmented
   - Use process separation (automatic with CLI)
   - First run downloads models (subsequent runs are faster)

## Troubleshooting

### Common Issues

1. **"Model not found"**
   - Models are auto-downloaded on first use
   - Check internet connection
   - Ensure enough disk space (~3GB for large-v3)

2. **Memory errors**
   - Use INT4 models
   - Reduce batch size
   - Close other applications

3. **Slow performance**
   - First run includes model download
   - Use smaller or INT4 models
   - Check Activity Monitor for other processes

### Getting Help

- Check the [MLX Backend Guide](../../MLX_BACKEND_GUIDE.md)
- Review [implementation status](../../MLX_IMPLEMENTATION_STATUS.md)
- Report issues on GitHub