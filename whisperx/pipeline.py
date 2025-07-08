"""
Unified pipeline interface for WhisperX
Implements the load_pipeline() API from the roadmap
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from whisperx.alignment import align, load_align_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.types import AlignedTranscriptionResult, TranscriptionResult
from whisperx.utils import get_writer
from whisperx.vads import load_vad_model


@dataclass
class PipelineConfig:
    """Configuration for the unified pipeline."""
    vad_filter: bool = True
    vad_method: str = "silero"
    vad_options: Optional[Dict] = None
    align_model: Optional[str] = None
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = None
    device: str = "cpu"
    compute_type: str = "float16"
    threads: int = 0


class UnifiedPipeline:
    """Unified WhisperX pipeline as specified in the roadmap."""
    
    def __init__(
        self,
        backend,  # ASR backend (MLX, FasterWhisper, etc.)
        vad_filter: bool = True,
        vad_method: str = "silero",
        vad_options: Optional[Dict] = None,
        align_model: Optional[str] = None,
        diarize: bool = False,
        diarize_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        """Initialize unified pipeline.
        
        Args:
            backend: ASR backend instance (e.g., MlxWhisper)
            vad_filter: Whether to use VAD filtering
            vad_method: VAD method ("silero" or "pyannote")
            vad_options: VAD configuration options
            align_model: Language code for alignment model (e.g., "en")
            diarize: Whether to perform speaker diarization
            diarize_model: Diarization model name
            hf_token: HuggingFace token for gated models
            device: Device for PyTorch models
        """
        self.backend = backend
        self.config = PipelineConfig(
            vad_filter=vad_filter,
            vad_method=vad_method,
            vad_options=vad_options or {},
            align_model=align_model,
            diarize=diarize,
            diarize_model=diarize_model,
            hf_token=hf_token,
            device=device,
            **kwargs
        )
        
        # Lazy-load components
        self._vad_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None
        
        # Set PyTorch to CPU-only as per roadmap
        if self.config.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_num_threads(kwargs.get("threads", 0) or 4)
    
    def _load_vad_model(self):
        """Lazy-load VAD model."""
        if self._vad_model is None and self.config.vad_filter:
            self._vad_model = load_vad_model(
                self.config.vad_method,
                device=torch.device(self.config.device),
                **self.config.vad_options
            )
        return self._vad_model
    
    def _load_align_model(self, language: str):
        """Lazy-load alignment model."""
        if self._align_model is None and self.config.align_model:
            self._align_model, self._align_metadata = load_align_model(
                language,
                device=self.config.device,
                model_name=self.config.align_model
            )
        return self._align_model, self._align_metadata
    
    def _load_diarize_model(self):
        """Lazy-load diarization model."""
        if self._diarize_model is None and self.config.diarize:
            self._diarize_model = DiarizationPipeline(
                model_name=self.config.diarize_model,
                use_auth_token=self.config.hf_token,
                device=torch.device(self.config.device)
            )
        return self._diarize_model
    
    def __call__(
        self,
        audio: Union[str, np.ndarray],
        batch_size: int = 8,
        chunk_size: int = 30,
        print_progress: bool = False,
        return_char_alignments: bool = False,
        **kwargs
    ) -> Dict:
        """Process audio through the full pipeline.
        
        Args:
            audio: Audio file path or numpy array
            batch_size: Batch size for processing
            chunk_size: Maximum chunk size in seconds
            print_progress: Whether to print progress
            return_char_alignments: Return character-level alignments
            
        Returns:
            Dictionary with segments, language, and optional word/speaker info
        """
        # Load audio if path
        if isinstance(audio, str):
            audio_path = audio
            audio_array = load_audio(audio_path)
        else:
            audio_path = None
            audio_array = audio
        
        # Stage 1: VAD filtering (optional)
        if self.config.vad_filter:
            if print_progress:
                print(">>Performing voice activity detection...")
            
            vad_model = self._load_vad_model()
            vad_segments = self._run_vad(audio_array, vad_model)
            
            if print_progress:
                print(f"VAD: {len(vad_segments)} speech segments")
        else:
            # No VAD - process entire audio
            vad_segments = [{"start": 0, "end": len(audio_array) / 16000}]
        
        # Stage 2: ASR with backend
        if print_progress:
            print(">>Performing speech recognition...")
        
        result = self._run_asr(
            audio_array,
            vad_segments,
            batch_size=batch_size,
            print_progress=print_progress,
            **kwargs
        )
        
        # Stage 3: Forced alignment (optional)
        if self.config.align_model and len(result["segments"]) > 0:
            if print_progress:
                print(">>Performing forced alignment...")
            
            # Detect language if needed
            language = result.get("language", "en")
            if self.config.align_model == "auto":
                align_language = language
            else:
                align_language = self.config.align_model
            
            # Load alignment model
            align_model, align_metadata = self._load_align_model(align_language)
            
            if align_model is not None:
                # Run alignment
                result = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio_array,
                    self.config.device,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )
        
        # Stage 4: Speaker diarization (optional)
        if self.config.diarize and len(result["segments"]) > 0:
            if print_progress:
                print(">>Performing speaker diarization...")
            
            diarize_model = self._load_diarize_model()
            
            # Run diarization
            if audio_path:
                diarize_segments = diarize_model(audio_path)
            else:
                # Save temp file for diarization
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    import soundfile as sf
                    sf.write(f.name, audio_array, 16000)
                    diarize_segments = diarize_model(f.name)
                    os.unlink(f.name)
            
            # Assign speakers to words
            result = assign_word_speakers(diarize_segments, result)
        
        return result
    
    def _run_vad(self, audio: np.ndarray, vad_model) -> List[Dict]:
        """Run VAD on audio."""
        # Prepare audio for VAD
        vad_input = {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": 16000
        }
        
        # Run VAD
        speech_timestamps = vad_model(vad_input)
        
        # Merge chunks
        merged = vad_model.merge_chunks(
            speech_timestamps,
            chunk_size=self.config.vad_options.get("chunk_size", 30)
        )
        
        # Convert to dict format
        segments = []
        for seg in merged:
            segments.append({
                "start": seg["start"],
                "end": seg["end"]
            })
        
        return segments
    
    def _run_asr(
        self,
        audio: np.ndarray,
        vad_segments: List[Dict],
        batch_size: int = 8,
        print_progress: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """Run ASR on VAD segments."""
        all_segments = []
        
        # Process segments in batches
        for i in range(0, len(vad_segments), batch_size):
            batch = vad_segments[i:i + batch_size]
            
            if print_progress:
                print(f"Processing batch {i//batch_size + 1}/{(len(vad_segments) + batch_size - 1)//batch_size}")
            
            # Extract audio for each segment
            batch_audio = []
            for seg in batch:
                start_sample = int(seg["start"] * 16000)
                end_sample = int(seg["end"] * 16000)
                segment_audio = audio[start_sample:end_sample]
                batch_audio.append(segment_audio)
            
            # Transcribe batch with backend
            if hasattr(self.backend, 'transcribe_batch'):
                # Backend supports batching
                batch_results = self.backend.transcribe_batch(batch_audio, **kwargs)
            else:
                # Process sequentially
                batch_results = []
                for segment_audio in batch_audio:
                    result = self.backend.transcribe(segment_audio, **kwargs)
                    batch_results.append(result)
            
            # Merge results
            for seg, result in zip(batch, batch_results):
                if "segments" in result:
                    # Adjust timestamps
                    for s in result["segments"]:
                        s["start"] += seg["start"]
                        s["end"] += seg["start"]
                        all_segments.append(s)
                else:
                    # Simple format
                    all_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": result.get("text", "")
                    })
        
        # Get language from first result
        language = None
        if batch_results and "language" in batch_results[0]:
            language = batch_results[0]["language"]
        
        return {
            "segments": all_segments,
            "language": language or "en"
        }
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up models
        del self._vad_model
        del self._align_model
        del self._diarize_model
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_pipeline(
    backend,
    vad_filter: bool = True,
    align_model: Optional[str] = None,
    diarize: bool = False,
    **kwargs
) -> UnifiedPipeline:
    """Load a unified WhisperX pipeline.
    
    This is the main entry point matching the roadmap specification.
    
    Args:
        backend: ASR backend instance (e.g., MlxWhisper)
        vad_filter: Whether to use VAD filtering
        align_model: Language code for alignment (e.g., "en")
        diarize: Whether to perform speaker diarization
        **kwargs: Additional pipeline options
        
    Returns:
        UnifiedPipeline instance
        
    Example:
        >>> from whisperx.backends.mlx_whisper import MlxWhisper
        >>> asr = MlxWhisper("~/mlx_models/large-v3-int4", batch_size=8)
        >>> pipe = whisperx.load_pipeline(
        ...     backend=asr,
        ...     vad_filter=True,
        ...     align_model="en",
        ...     diarize=True
        ... )
        >>> result = pipe("input.wav")
    """
    return UnifiedPipeline(
        backend=backend,
        vad_filter=vad_filter,
        align_model=align_model,
        diarize=diarize,
        **kwargs
    )


# Convenience function for creating MLX pipeline
def load_mlx_pipeline(
    model_path: str = "mlx-community/whisper-large-v3",
    batch_size: int = 8,
    dtype: str = "float16",
    vad_filter: bool = True,
    align_model: Optional[str] = None,
    diarize: bool = False,
    **kwargs
) -> UnifiedPipeline:
    """Create a pipeline with MLX backend.
    
    Convenience function that creates the MLX backend and pipeline.
    
    Args:
        model_path: Path or HF repo for MLX model
        batch_size: Batch size for processing
        dtype: Model precision ("float16" or "float32")
        vad_filter: Whether to use VAD
        align_model: Language for alignment
        diarize: Whether to diarize
        
    Returns:
        UnifiedPipeline with MLX backend
    """
    from whisperx.backends.mlx_whisper import MlxWhisper
    
    backend = MlxWhisper(
        model_path=model_path,
        batch_size=batch_size,
        dtype=dtype,
        **kwargs
    )
    
    return load_pipeline(
        backend=backend,
        vad_filter=vad_filter,
        align_model=align_model,
        diarize=diarize,
        **kwargs
    )