"""
WhisperX ASR Module - MLX Backend Only

This module provides the main ASR interface for WhisperX using only the MLX backend.
Supports both standard and batch-optimized processing.
"""
import os
import platform
from typing import Optional, Union, Dict, Any, List

import numpy as np
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, SingleSegment
from whisperx.vads import Vad, Silero, Pyannote


class MLXWhisperPipeline:
    """
    Pipeline wrapper for MLX Whisper model to maintain API compatibility.
    Handles VAD segmentation and dispatches to appropriate backend.
    """
    
    def __init__(
        self,
        backend,
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
        Transcribe audio using MLX backend with optional VAD segmentation.
        
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
            
        # Load audio if needed
        if isinstance(audio, str):
            audio = load_audio(audio)
            
        # If VAD is provided, segment the audio first
        if self.vad_model is not None:
            # Run VAD to get segments
            segments = self._segment_audio_with_vad(audio, chunk_size)
            
            # Check if backend supports batch processing
            if hasattr(self.backend, 'transcribe_batch'):
                # Use batch processing backend
                options = {
                    'language': language,
                    'task': task,
                    'print_progress': print_progress,
                    'suppress_numerals': self.suppress_numerals,
                    'batch_size': batch_size,
                }
                options.update(transcribe_options)
                
                # Process segments in batches
                all_segments = []
                for i in range(0, len(segments), batch_size):
                    batch = segments[i:i + batch_size]
                    batch_results = self.backend.transcribe_batch(
                        batch,
                        **options
                    )
                    
                    # Convert batch results to segments
                    for seg_input, result in zip(batch, batch_results):
                        if result.get('segments'):
                            # Adjust timestamps relative to segment start
                            for seg in result['segments']:
                                seg['start'] += seg_input['start']
                                seg['end'] += seg_input['start']
                                all_segments.append(SingleSegment(**seg))
                        else:
                            # Create single segment from result
                            segment = SingleSegment(
                                start=seg_input['start'],
                                end=seg_input['end'],
                                text=result.get('text', '').strip()
                            )
                            all_segments.append(segment)
                
                # Get language from first result
                detected_language = batch_results[0].get('language', language) if batch_results else language
                
                return TranscriptionResult(
                    segments=all_segments,
                    language=detected_language
                )
            else:
                # Backend doesn't support batch processing, use standard transcribe
                # with VAD segments already prepared
                options = {
                    'language': language,
                    'task': task,
                    'chunk_size': chunk_size,
                    'print_progress': print_progress,
                    'combined_progress': combined_progress,
                    'suppress_numerals': self.suppress_numerals,
                    'vad_segments': [{'start': s['start'], 'end': s['end']} for s in segments],
                }
                options.update(transcribe_options)
                return self.backend.transcribe(audio, **options)
        else:
            # No VAD, use backend's transcribe directly
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
            return self.backend.transcribe(audio, **options)
            
    def _segment_audio_with_vad(self, audio: np.ndarray, chunk_size: int) -> List[Dict[str, Any]]:
        """Run VAD and return audio segments."""
        # Pre-process audio
        if hasattr(self.vad_model, 'preprocess_audio'):
            waveform = self.vad_model.preprocess_audio(audio)
        else:
            # Default preprocessing
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
        # Run VAD
        vad_result = self.vad_model(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE},
            **self._vad_params
        )
        
        # Merge chunks
        if hasattr(self.vad_model, 'merge_chunks'):
            vad_segments = self.vad_model.merge_chunks(
                vad_result,
                chunk_size,
                onset=self._vad_params.get('vad_onset', 0.5),
                offset=self._vad_params.get('vad_offset', 0.363),
            )
        else:
            vad_segments = vad_result
            
        # Create segments with audio data
        segments = []
        for vad_seg in vad_segments:
            start_sample = int(vad_seg['start'] * SAMPLE_RATE)
            end_sample = int(vad_seg['end'] * SAMPLE_RATE)
            
            segment = {
                'start': vad_seg['start'],
                'end': vad_seg['end'],
                'audio': audio[start_sample:end_sample]
            }
            segments.append(segment)
            
        return segments
        
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
    backend: str = "auto",  # Choose backend: "auto", "standard", "batch"
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
        backend: Backend to use - "auto", "standard", "batch"
        
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
            # Ensure required Silero parameters
            vad_kwargs.setdefault('vad_onset', 0.5)
            vad_kwargs.setdefault('vad_offset', 0.363)
            vad_kwargs.setdefault('chunk_size', 30)
            vad_model = Silero(**vad_kwargs)
        elif vad_method == "pyannote":
            # Apply PyAnnote patch for PyTorch 2.6+ compatibility
            try:
                from whisperx.vads.pyannote_patch import patch_pyannote
                patch_pyannote()
            except ImportError:
                pass
            # Ensure required PyAnnote parameters
            # For MLX, we use CPU for VAD since MLX handles GPU acceleration differently
            vad_kwargs.setdefault('device', torch.device('cpu'))
            vad_kwargs.setdefault('vad_onset', 0.5)
            vad_kwargs.setdefault('vad_offset', 0.363)
            vad_model = Pyannote(**vad_kwargs)
        else:
            vad_model = None
            
    # Determine which backend to use
    batch_size = asr_options.get('batch_size', 8) if asr_options else 8
    use_batch_backend = backend == "batch" or (backend == "auto" and batch_size > 1 and vad_model is not None)
    
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
        'batch_size': batch_size,
    }
    
    # Initialize appropriate MLX backend
    if use_batch_backend:
        # Use optimized batch backend for parallel processing
        from whisperx.backends.mlx_batch_optimized import OptimizedBatchMLXWhisperBackend
        backend_instance = OptimizedBatchMLXWhisperBackend(
            model_name=whisper_arch,
            batch_size=batch_size,
            compute_type=compute_type,
            asr_options=asr_options,
            quantization="int4" if compute_type == "int4" else None,
            model_path=download_root
        )
    else:
        # Use standard MLX backend with internal VAD handling
        from whisperx.backends.mlx_whisper import MlxWhisperBackend
        backend_instance = MlxWhisperBackend(**backend_options)
    
    # Create pipeline
    pipeline_options = {
        'language': language,
        'suppress_numerals': asr_options.get('suppress_numerals', False) if asr_options else False,
        'batch_size': batch_size,
    }
    
    return MLXWhisperPipeline(
        backend=backend_instance,
        vad=vad_model,
        vad_params=vad_options or {},
        **pipeline_options
    )


# Re-export for compatibility
__all__ = ['load_model', 'MLXWhisperPipeline']