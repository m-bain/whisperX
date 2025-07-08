"""
Batch processing implementation for WhisperX MLX backend
Implements efficient batching for 30s chunks as per roadmap
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from whisperx.audio import SAMPLE_RATE


@dataclass
class AudioChunk:
    """Represents a chunk of audio for batch processing."""
    audio: np.ndarray
    start_time: float
    end_time: float
    segment_idx: int
    

class BatchProcessor:
    """Handles efficient batch processing of audio segments."""
    
    def __init__(
        self,
        batch_size: int = 8,
        chunk_duration: float = 30.0,
        overlap: float = 0.5,
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of chunks to process in parallel
            chunk_duration: Maximum duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
        """
        self.batch_size = batch_size
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.chunk_samples = int(chunk_duration * SAMPLE_RATE)
        self.overlap_samples = int(overlap * SAMPLE_RATE)
    
    def create_chunks(
        self,
        audio: np.ndarray,
        segments: List[Dict[str, float]]
    ) -> List[AudioChunk]:
        """Create chunks from audio segments.
        
        Args:
            audio: Full audio array
            segments: List of segments with start/end times
            
        Returns:
            List of AudioChunk objects
        """
        chunks = []
        
        for idx, segment in enumerate(segments):
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            
            if duration <= self.chunk_duration:
                # Single chunk for this segment
                start_sample = int(start_time * SAMPLE_RATE)
                end_sample = int(end_time * SAMPLE_RATE)
                
                chunk = AudioChunk(
                    audio=audio[start_sample:end_sample],
                    start_time=start_time,
                    end_time=end_time,
                    segment_idx=idx
                )
                chunks.append(chunk)
            else:
                # Split into multiple chunks with overlap
                num_chunks = math.ceil(duration / (self.chunk_duration - self.overlap))
                
                for i in range(num_chunks):
                    chunk_start = start_time + i * (self.chunk_duration - self.overlap)
                    chunk_end = min(chunk_start + self.chunk_duration, end_time)
                    
                    start_sample = int(chunk_start * SAMPLE_RATE)
                    end_sample = int(chunk_end * SAMPLE_RATE)
                    
                    chunk = AudioChunk(
                        audio=audio[start_sample:end_sample],
                        start_time=chunk_start,
                        end_time=chunk_end,
                        segment_idx=idx
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def create_batches(self, chunks: List[AudioChunk]) -> List[List[AudioChunk]]:
        """Group chunks into batches.
        
        Args:
            chunks: List of audio chunks
            
        Returns:
            List of batches
        """
        batches = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def pad_batch(self, batch: List[AudioChunk]) -> Tuple[np.ndarray, List[int]]:
        """Pad audio chunks in batch to same length.
        
        Args:
            batch: List of audio chunks
            
        Returns:
            Padded audio array and original lengths
        """
        # Find maximum length in batch
        max_length = max(len(chunk.audio) for chunk in batch)
        
        # Pad all chunks to same length
        padded_audio = []
        original_lengths = []
        
        for chunk in batch:
            audio = chunk.audio
            original_lengths.append(len(audio))
            
            if len(audio) < max_length:
                # Pad with zeros
                padding = max_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            padded_audio.append(audio)
        
        # Stack into batch array
        batch_audio = np.stack(padded_audio)
        
        return batch_audio, original_lengths
    
    def process_batch_with_mlx(
        self,
        batch: List[AudioChunk],
        model_path: str,
        decode_options: Dict
    ) -> List[Dict]:
        """Process a batch of chunks using MLX.
        
        Args:
            batch: List of audio chunks
            model_path: Path to MLX model
            decode_options: Decoding options
            
        Returns:
            List of transcription results
        """
        import mlx_whisper
        
        # Pad batch to same length
        batch_audio, original_lengths = self.pad_batch(batch)
        
        # Process each chunk (MLX doesn't support batching yet)
        # In future, this could be: results = mlx_whisper.transcribe_batch(...)
        results = []
        
        for i, chunk in enumerate(batch):
            # Use original audio without padding
            result = mlx_whisper.transcribe(
                chunk.audio,
                path_or_hf_repo=model_path,
                **decode_options
            )
            results.append(result)
        
        return results
    
    def merge_results(
        self,
        chunks: List[AudioChunk],
        results: List[Dict],
        segments: List[Dict]
    ) -> List[Dict]:
        """Merge chunk results back into segments.
        
        Args:
            chunks: Original chunks
            results: Transcription results
            segments: Original segments
            
        Returns:
            Updated segments with transcriptions
        """
        # Group results by segment index
        segment_results = {}
        
        for chunk, result in zip(chunks, results):
            idx = chunk.segment_idx
            if idx not in segment_results:
                segment_results[idx] = []
            segment_results[idx].append((chunk, result))
        
        # Merge results for each segment
        final_segments = []
        
        for idx, segment in enumerate(segments):
            if idx not in segment_results:
                # No results for this segment
                final_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": ""
                })
                continue
            
            # Get all chunk results for this segment
            chunk_results = segment_results[idx]
            
            if len(chunk_results) == 1:
                # Single chunk - use result directly
                _, result = chunk_results[0]
                text = result.get("text", "").strip()
            else:
                # Multiple chunks - merge with overlap handling
                text = self._merge_overlapping_text(chunk_results)
            
            final_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text
            })
        
        return final_segments
    
    def _merge_overlapping_text(
        self,
        chunk_results: List[Tuple[AudioChunk, Dict]]
    ) -> str:
        """Merge text from overlapping chunks.
        
        Simple strategy: concatenate non-overlapping portions.
        Future: implement more sophisticated merging.
        """
        if not chunk_results:
            return ""
        
        # Sort by start time
        chunk_results.sort(key=lambda x: x[0].start_time)
        
        # Take first chunk's text as base
        merged_text = chunk_results[0][1].get("text", "").strip()
        
        # Add non-overlapping portions from subsequent chunks
        for i in range(1, len(chunk_results)):
            chunk, result = chunk_results[i]
            text = result.get("text", "").strip()
            
            # Simple heuristic: take last 20% of words as potential overlap
            words = text.split()
            if len(words) > 5:
                # Skip first 20% of words (likely overlap)
                skip = len(words) // 5
                text = " ".join(words[skip:])
            
            if text:
                merged_text += " " + text
        
        return merged_text


def batch_transcribe(
    audio: np.ndarray,
    segments: List[Dict],
    model_path: str,
    batch_size: int = 8,
    chunk_duration: float = 30.0,
    decode_options: Optional[Dict] = None,
    print_progress: bool = False,
) -> List[Dict]:
    """Transcribe segments using batch processing.
    
    Args:
        audio: Full audio array
        segments: VAD segments
        model_path: Path to MLX model
        batch_size: Batch size
        chunk_duration: Maximum chunk duration
        decode_options: Decoding options
        print_progress: Whether to print progress
        
    Returns:
        Transcribed segments
    """
    processor = BatchProcessor(
        batch_size=batch_size,
        chunk_duration=chunk_duration,
        overlap=0.5
    )
    
    # Create chunks
    chunks = processor.create_chunks(audio, segments)
    
    if print_progress:
        print(f"Created {len(chunks)} chunks from {len(segments)} segments")
    
    # Create batches
    batches = processor.create_batches(chunks)
    
    if print_progress:
        print(f"Processing {len(batches)} batches of size {batch_size}")
    
    # Process batches
    all_results = []
    
    for i, batch in enumerate(batches):
        if print_progress:
            print(f"Processing batch {i+1}/{len(batches)}")
        
        batch_results = processor.process_batch_with_mlx(
            batch,
            model_path,
            decode_options or {}
        )
        
        all_results.extend(batch_results)
    
    # Merge results
    final_segments = processor.merge_results(chunks, all_results, segments)
    
    return final_segments


# Memory optimization utilities
def optimize_memory_mlx():
    """Apply memory optimization settings for MLX."""
    # Set maximum workspace for Metal compilation
    # This constrains the Metal tiling scratchpad
    mx.metal.set_memory_limit(4 * 1024 * 1024 * 1024)  # 4GB
    
    # Clear cache before processing
    mx.clear_cache()


def move_to_cpu_when_idle(array: mx.array) -> mx.array:
    """Move MLX array to CPU when idle to save GPU memory."""
    # Force evaluation first
    mx.eval(array)
    
    # Move to CPU
    cpu_array = mx.array(array, dtype=array.dtype)
    
    # Clear GPU cache
    mx.clear_cache()
    
    return cpu_array


class MemoryEfficientProcessor:
    """Memory-efficient batch processor for 8GB devices."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        """Initialize with memory constraint."""
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        optimize_memory_mlx()
    
    def process_with_memory_limit(
        self,
        audio: np.ndarray,
        segments: List[Dict],
        model_path: str,
        batch_size: int = 4,  # Smaller batch for memory
    ) -> List[Dict]:
        """Process with strict memory limits."""
        # Use smaller batches and clear cache frequently
        processor = BatchProcessor(
            batch_size=batch_size,
            chunk_duration=20.0,  # Shorter chunks
            overlap=0.3
        )
        
        chunks = processor.create_chunks(audio, segments)
        batches = processor.create_batches(chunks)
        
        all_results = []
        
        for i, batch in enumerate(batches):
            # Clear cache before each batch
            mx.clear_cache()
            
            # Process batch
            batch_results = processor.process_batch_with_mlx(
                batch,
                model_path,
                {"temperature": 0.0}  # Greedy for memory efficiency
            )
            
            all_results.extend(batch_results)
            
            # Clear cache after each batch
            mx.clear_cache()
            
            # Check memory usage
            if (i + 1) % 5 == 0:
                import psutil
                process = psutil.Process()
                mem_usage = process.memory_info().rss
                if mem_usage > self.max_memory_bytes:
                    print(f"Warning: Memory usage {mem_usage / 1024**3:.1f}GB exceeds limit")
        
        # Final merge
        final_segments = processor.merge_results(chunks, all_results, segments)
        
        # Final cleanup
        mx.clear_cache()
        
        return final_segments