#!/usr/bin/env python3
"""
Unit tests for WhisperX MLX backend
As specified in the roadmap: ~10s runtime, uses 15s WAV
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import whisperx
from whisperx.backends.mlx_whisper_v2 import MlxWhisper, MlxWhisperBackend
from whisperx.pipeline import load_pipeline, load_mlx_pipeline


class TestMLXBackend(unittest.TestCase):
    """Test suite for MLX backend implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Create test audio files."""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create 15-second test audio as per roadmap
        cls.test_audio_15s = os.path.join(cls.test_dir, "test_15s.wav")
        cls._create_test_audio(cls.test_audio_15s, duration=15)
        
        # Create short test audio for quick tests
        cls.test_audio_short = os.path.join(cls.test_dir, "test_short.wav")
        cls._create_test_audio(cls.test_audio_short, duration=3)
    
    @staticmethod
    def _create_test_audio(filepath: str, duration: int):
        """Create test audio file using macOS say command."""
        import subprocess
        
        text = "This is a test of the WhisperX MLX backend. " * (duration // 3)
        
        subprocess.run([
            "say", "-o", filepath,
            "--data-format=LEF32@16000",
            text[:500]  # Limit text length
        ], capture_output=True)
    
    def test_mlx_whisper_init(self):
        """Test MlxWhisper initialization matching roadmap API."""
        # Test initialization as per roadmap
        asr = MlxWhisper(
            model_path="mlx-community/whisper-tiny",
            batch_size=8,
            dtype="float16",
            word_timestamps=True,
            language="en"
        )
        
        self.assertEqual(asr.model_path, "mlx-community/whisper-tiny")
        self.assertEqual(asr.batch_size, 8)
        self.assertEqual(asr.dtype, "float16")
        self.assertTrue(asr.word_timestamps)
        self.assertEqual(asr.language, "en")
    
    def test_mlx_backend_init(self):
        """Test MlxWhisperBackend initialization."""
        backend = MlxWhisperBackend(
            model="tiny",
            device="mlx",
            compute_type="float16",
            batch_size=8,
            word_timestamps=True
        )
        
        # Check model name mapping
        self.assertEqual(backend.path_or_hf_repo, "mlx-community/whisper-tiny")
        self.assertEqual(backend.batch_size, 8)
        self.assertTrue(backend.word_timestamps)
    
    def test_model_name_mapping(self):
        """Test model name mapping for all variants."""
        backend = MlxWhisperBackend("tiny", device="mlx")
        
        # Test standard models
        test_cases = [
            ("tiny", "mlx-community/whisper-tiny"),
            ("base", "mlx-community/whisper-base"),
            ("small", "mlx-community/whisper-small"),
            ("large-v3", "mlx-community/whisper-large-v3"),
            ("tiny-q4", "mlx-community/whisper-tiny-mlx-q4"),
            ("large-v3-q4", "mlx-community/whisper-large-v3-mlx-q4"),
        ]
        
        for short_name, expected in test_cases:
            backend = MlxWhisperBackend(short_name)
            self.assertEqual(backend.path_or_hf_repo, expected)
    
    def test_basic_transcription(self):
        """Test basic transcription with tiny model."""
        backend = MlxWhisperBackend(
            model="tiny",
            compute_type="float16",
            word_timestamps=False
        )
        
        # Transcribe short audio
        result = backend.transcribe(
            self.test_audio_short,
            language="en",
            task="transcribe"
        )
        
        # Check result structure
        self.assertIn("segments", result)
        self.assertIn("language", result)
        self.assertEqual(result["language"], "en")
        self.assertIsInstance(result["segments"], list)
        
        # Check segments
        if result["segments"]:
            segment = result["segments"][0]
            self.assertIn("start", segment)
            self.assertIn("end", segment)
            self.assertIn("text", segment)
            self.assertIsInstance(segment["text"], str)
    
    def test_word_timestamps(self):
        """Test word timestamp extraction."""
        backend = MlxWhisperBackend(
            model="tiny",
            word_timestamps=True
        )
        
        # Transcribe with word timestamps
        result = backend.transcribe(
            self.test_audio_short,
            language="en",
            word_timestamps=True
        )
        
        # Check for word timestamps in segments
        if result["segments"] and "words" in result["segments"][0]:
            words = result["segments"][0]["words"]
            self.assertIsInstance(words, list)
            if words:
                word = words[0]
                self.assertIn("word", word)
                self.assertIn("start", word)
                self.assertIn("end", word)
    
    def test_language_detection(self):
        """Test automatic language detection."""
        backend = MlxWhisperBackend("tiny")
        
        # Detect language
        detected = backend.detect_language(self.test_audio_short)
        
        # Should detect English
        self.assertEqual(detected, "en")
    
    def test_temperature_handling(self):
        """Test temperature parameter handling."""
        backend = MlxWhisperBackend(
            model="tiny",
            options={"temperatures": [0.0, 0.2, 0.4]}
        )
        
        # Should handle temperature list
        result = backend.transcribe(
            self.test_audio_short,
            language="en"
        )
        
        self.assertIn("segments", result)
    
    def test_unified_pipeline(self):
        """Test unified pipeline interface from roadmap."""
        # Create MLX backend
        asr = MlxWhisper(
            model_path="mlx-community/whisper-tiny",
            batch_size=8,
            word_timestamps=True,
            language="en"
        )
        
        # Create unified pipeline
        pipe = load_pipeline(
            backend=asr.backend,
            vad_filter=True,
            align_model=None,  # No alignment for MLX yet
            diarize=False
        )
        
        # Process audio
        result = pipe(self.test_audio_short)
        
        # Check result
        self.assertIn("segments", result)
        self.assertIn("language", result)
    
    def test_mlx_pipeline_convenience(self):
        """Test convenience function for MLX pipeline."""
        pipe = load_mlx_pipeline(
            model_path="mlx-community/whisper-tiny",
            batch_size=8,
            vad_filter=True,
            align_model=None,
            diarize=False
        )
        
        # Process audio
        result = pipe(self.test_audio_short)
        
        # Check result
        self.assertIn("segments", result)
        self.assertIn("language", result)
    
    def test_batch_transcription(self):
        """Test batch transcription capability."""
        backend = MlxWhisperBackend(
            model="tiny",
            batch_size=2
        )
        
        # Create multiple audio segments
        from whisperx.audio import load_audio
        audio = load_audio(self.test_audio_short)
        
        # Split into 2 segments
        mid = len(audio) // 2
        segments = [audio[:mid], audio[mid:]]
        
        # Transcribe batch
        results = backend.transcribe_batch(
            segments,
            language="en"
        )
        
        # Check results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("segments", result)
            self.assertIn("language", result)
    
    def test_performance_benchmark(self):
        """Test performance meets roadmap targets."""
        backend = MlxWhisperBackend("tiny")
        
        # Load 15s audio
        from whisperx.audio import load_audio
        audio = load_audio(self.test_audio_15s)
        duration = len(audio) / 16000
        
        # Warm-up
        _ = backend.transcribe(audio[:16000], language="en")
        
        # Benchmark
        start = time.perf_counter()
        result = backend.transcribe(audio, language="en")
        elapsed = time.perf_counter() - start
        
        # Calculate RTF
        rtf = duration / elapsed
        
        print(f"\nPerformance: {rtf:.1f}x real-time")
        
        # Should exceed 20x for tiny model (roadmap target)
        self.assertGreater(rtf, 20.0, f"RTF {rtf} < 20x target")
    
    def test_int4_model_support(self):
        """Test INT4 quantized model support."""
        backend = MlxWhisperBackend(
            model="tiny-q4",  # INT4 model
            compute_type="int4"
        )
        
        # Check model path mapping
        self.assertEqual(
            backend.path_or_hf_repo,
            "mlx-community/whisper-tiny-mlx-q4"
        )
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        from whisperx.batch_processor import optimize_memory_mlx
        
        # Should not raise
        optimize_memory_mlx()
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        backend = MlxWhisperBackend("tiny")
        
        # Test with invalid audio path
        with self.assertRaises(Exception):
            backend.transcribe("/nonexistent/audio.wav")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)


class TestBatchProcessor(unittest.TestCase):
    """Test batch processing functionality."""
    
    def test_chunk_creation(self):
        """Test audio chunk creation."""
        from whisperx.batch_processor import BatchProcessor, AudioChunk
        
        processor = BatchProcessor(
            batch_size=4,
            chunk_duration=30.0,
            overlap=0.5
        )
        
        # Create test segments
        segments = [
            {"start": 0.0, "end": 10.0},
            {"start": 10.0, "end": 50.0},  # Longer than chunk_duration
        ]
        
        # Create test audio
        audio = np.zeros(60 * 16000)  # 60 seconds
        
        # Create chunks
        chunks = processor.create_chunks(audio, segments)
        
        # First segment should create 1 chunk
        # Second segment should create 2 chunks (40s > 30s)
        self.assertGreaterEqual(len(chunks), 3)
        
        # Check chunk properties
        for chunk in chunks:
            self.assertIsInstance(chunk, AudioChunk)
            self.assertIsInstance(chunk.audio, np.ndarray)
            self.assertGreaterEqual(chunk.start_time, 0)
            self.assertGreater(chunk.end_time, chunk.start_time)
    
    def test_batch_creation(self):
        """Test batch grouping."""
        from whisperx.batch_processor import BatchProcessor, AudioChunk
        
        processor = BatchProcessor(batch_size=3)
        
        # Create test chunks
        chunks = [
            AudioChunk(np.zeros(1000), 0, 1, 0),
            AudioChunk(np.zeros(1000), 1, 2, 0),
            AudioChunk(np.zeros(1000), 2, 3, 0),
            AudioChunk(np.zeros(1000), 3, 4, 0),
            AudioChunk(np.zeros(1000), 4, 5, 0),
        ]
        
        # Create batches
        batches = processor.create_batches(chunks)
        
        # Should create 2 batches (3 + 2)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 2)
    
    def test_padding(self):
        """Test batch padding to same length."""
        from whisperx.batch_processor import BatchProcessor, AudioChunk
        
        processor = BatchProcessor()
        
        # Create chunks with different lengths
        chunks = [
            AudioChunk(np.ones(1000), 0, 1, 0),
            AudioChunk(np.ones(1500), 1, 2, 0),
            AudioChunk(np.ones(800), 2, 3, 0),
        ]
        
        # Pad batch
        padded, lengths = processor.pad_batch(chunks)
        
        # Check padding
        self.assertEqual(padded.shape, (3, 1500))  # Max length
        self.assertEqual(lengths, [1000, 1500, 800])
        
        # Check padded values
        self.assertTrue(np.all(padded[0, :1000] == 1))  # Original data
        self.assertTrue(np.all(padded[0, 1000:] == 0))  # Padding


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_mlx(self):
        """Test full pipeline with MLX backend."""
        # Create test audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            test_file = f.name
            TestMLXBackend._create_test_audio(test_file, duration=5)
        
        try:
            # Run full pipeline
            pipe = load_mlx_pipeline(
                model_path="mlx-community/whisper-tiny",
                vad_filter=True,
                align_model=None,
                diarize=False
            )
            
            result = pipe(test_file, print_progress=False)
            
            # Check result
            self.assertIn("segments", result)
            self.assertIn("language", result)
            self.assertIsInstance(result["segments"], list)
            
        finally:
            os.unlink(test_file)


def run_tests():
    """Run all tests and report timing."""
    start = time.time()
    
    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    elapsed = time.time() - start
    print(f"\n✓ Tests completed in {elapsed:.1f}s")
    
    # Check ~10s target from roadmap
    if elapsed < 15:
        print("✓ Meets roadmap target of ~10s runtime")
    else:
        print(f"⚠️  Exceeds roadmap target ({elapsed:.1f}s > 10s)")


if __name__ == "__main__":
    run_tests()