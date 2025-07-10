#!/bin/bash
# Test script to verify fresh clone functionality

echo "=== Testing Fresh Clone Functionality ==="
echo "This script will:"
echo "1. Clone the MLX fork to a temporary directory"
echo "2. Install dependencies"
echo "3. Run basic tests"
echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Clone the repository
echo -e "\n1. Cloning repository..."
cd $TEMP_DIR
git clone https://github.com/sooth/whisperx-mlx.git
cd whisperx-mlx

# Install in development mode
echo -e "\n2. Installing dependencies..."
pip install -e .

# Create a simple test audio file
echo -e "\n3. Creating test audio..."
python -c "
import numpy as np
import wave
sr = 16000
t = np.linspace(0, 3, sr * 3)
signal = np.sin(2 * np.pi * 440 * t) * 0.5
with wave.open('test.wav', 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2) 
    w.setframerate(sr)
    w.writeframes((signal * 32767).astype(np.int16).tobytes())
print('Created test.wav')
"

# Test Python API
echo -e "\n4. Testing Python API..."
python -c "
import whisperx
print('Loading model...')
model = whisperx.load_model('tiny', device='cpu', backend='mlx')
print('Model loaded successfully!')

audio = whisperx.load_audio('test.wav')
print('Transcribing...')
result = model.transcribe(audio, batch_size=8)
print(f'Transcription complete! Segments: {len(result[\"segments\"])}')
"

# Test CLI
echo -e "\n5. Testing CLI..."
python -m whisperx test.wav --model tiny --backend mlx --output_format txt --no_align

# Check if output was created
if [ -f "test.txt" ]; then
    echo "✓ CLI test passed - output file created"
else
    echo "✗ CLI test failed - no output file"
fi

# Test batch backend
echo -e "\n6. Testing batch backend..."
python -m whisperx test.wav --model tiny --backend batch --output_format json --no_align

if [ -f "test.json" ]; then
    echo "✓ Batch backend test passed"
else
    echo "✗ Batch backend test failed"
fi

# Cleanup
echo -e "\n7. Cleaning up..."
cd /
rm -rf $TEMP_DIR

echo -e "\n=== Fresh Clone Test Complete ==="