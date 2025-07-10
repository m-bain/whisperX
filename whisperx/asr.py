"""
WhisperX ASR Module - MLX Backend Only

This module provides the main ASR interface for WhisperX using only the MLX backend.
"""
import os
import platform
from typing import Optional, Union, Dict, Any

import numpy as np
from transformers import Pipeline

from whisperx.audio import load_audio
from whisperx.backends.mlx_whisper import MlxWhisperBackend
from whisperx.types import TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote


class MLXWhisperPipeline(Pipeline):
    """
    Pipeline wrapper for MLX Whisper model to maintain API compatibility.
    """
    
    def __init__(
        self,
        backend: MlxWhisperBackend,
        vad: Optional[Vad] = None,
        vad_params: Optional[dict] = None,
        suppress_numerals: bool = False,
        **kwargs
    ):
        self.backend = backend
        self.model = backend  # Alias for compatibility
        self.vad_model = vad
        self._vad_params = vad_params or {}
        self.suppress_numerals = suppress_numerals
        self.preset_language = kwargs.get('language', None)
        self._batch_size = kwargs.get('batch_size', 8)
        
        # For Pipeline compatibility
        self.framework = "pt"
        self.device = "mlx"
        
        # Initialize parent without calling __init__ to avoid conflicts
        Pipeline.__bases__ = ()
        
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        **transcribe_options,
    ) -> TranscriptionResult:
        """
        Transcribe audio using MLX backend.
        
        Args:
            audio: Path to audio file or numpy array
            batch_size: Batch size for processing
            num_workers: Not used (for compatibility)
            language: Language code or None for auto-detection
            task: "transcribe" or "translate"
            chunk_size: Size of audio chunks in seconds
            print_progress: Whether to print progress
            combined_progress: Whether to combine progress bars
            **transcribe_options: Additional options for transcription
            
        Returns:
            TranscriptionResult with segments and detected language
        """
        # Override with pipeline settings if not specified
        if batch_size is None:
            batch_size = self._batch_size
        if language is None:
            language = self.preset_language
        if task is None:
            task = getattr(self.backend, 'task', 'transcribe')
            
        # Merge options
        options = {
            'batch_size': batch_size,
            'language': language,
            'task': task,
            'chunk_size': chunk_size,
            'print_progress': print_progress,
            'combined_progress': combined_progress,
            'suppress_numerals': self.suppress_numerals,
        }
        options.update(transcribe_options)
        
        # If VAD is provided, use it to segment audio first
        if self.vad_model is not None and isinstance(audio, (str, np.ndarray)):
            if isinstance(audio, str):
                audio_array = load_audio(audio)
            else:
                audio_array = audio
                
            # Run VAD
            vad_segments = self.vad_model(
                {"waveform": audio_array, "sample_rate": 16000},
                **self._vad_params
            )
            options['vad_segments'] = vad_segments
            
        # Call backend transcribe
        return self.backend.transcribe(audio, **options)
        
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect language of audio."""
        return self.backend.detect_language(audio)
        
    def __call__(self, *args, **kwargs):
        """Make pipeline callable."""
        return self.transcribe(*args, **kwargs)


def load_model(
    whisper_arch: str,
    device: str = "auto",
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model: Optional[Any] = None,  # Ignored - for compatibility
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,  # Ignored - MLX handles threading
) -> MLXWhisperPipeline:
    """
    Load a Whisper MLX model for inference.
    
    Args:
        whisper_arch: The name of the Whisper model to load (e.g., "base", "small", "large-v3")
        device: Device to use - ignored, MLX always uses Apple Silicon
        device_index: Device index - ignored for MLX
        compute_type: Compute type - "float16", "float32", "int8", "int4"
        asr_options: Additional ASR options
        language: The language to use for transcription
        vad_model: Pre-initialized VAD model
        vad_method: VAD method to use if vad_model is None
        vad_options: Options for VAD
        model: Pre-loaded model - ignored
        task: Task to perform - "transcribe" or "translate"
        download_root: Root directory for downloading models
        local_files_only: Whether to use only local files
        threads: Number of threads - ignored, MLX handles this
        
    Returns:
        MLXWhisperPipeline: Pipeline for transcription
    """
    # Check if running on Apple Silicon
    if platform.system() != "Darwin" or platform.processor() != "arm":
        raise RuntimeError("This MLX-only fork requires Apple Silicon (M1/M2/M3) hardware")
    
    # Initialize VAD if needed
    if vad_model is None and vad_method:
        vad_kwargs = vad_options or {}
        if vad_method == "silero":
            vad_model = Silero(**vad_kwargs)
        elif vad_method == "pyannote":
            # Apply PyAnnote patch for PyTorch 2.6+ compatibility
            try:
                from whisperx.vads.pyannote_patch import patch_pyannote
                patch_pyannote()
            except ImportError:
                pass
            vad_model = Pyannote(**vad_kwargs)
        else:
            vad_model = None
            
    # Prepare backend options
    backend_options = {
        'model': whisper_arch,
        'device': 'mlx',  # Always MLX
        'device_index': 0,  # Not used
        'compute_type': compute_type,
        'download_root': download_root,
        'local_files_only': local_files_only,
        'threads': threads,  # MLX will ignore this
        'asr_options': asr_options or {},
        'vad_method': vad_method,
        'vad_options': vad_options or {},
        'language': language,
        'task': task,
    }
    
    # Initialize MLX backend
    backend = MlxWhisperBackend(**backend_options)
    
    # Create pipeline
    pipeline_options = {
        'language': language,
        'suppress_numerals': asr_options.get('suppress_numerals', False) if asr_options else False,
        'batch_size': asr_options.get('batch_size', 8) if asr_options else 8,
    }
    
    return MLXWhisperPipeline(
        backend=backend,
        vad=vad_model,
        vad_params=vad_options or {},
        **pipeline_options
    )


# Re-export for compatibility
__all__ = ['load_model', 'MLXWhisperPipeline']