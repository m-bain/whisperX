"""
MLX Whisper backend for WhisperX - Corrected implementation
Uses proper mlx_whisper API with auto-download support
"""

import mlx.core as mx
import numpy as np
from typing import Optional, List, Dict, Union, Tuple
from whisperx.backends.base import WhisperBackend
from whisperx.types import TranscriptionResult, SingleSegment
from whisperx.audio import load_audio


class MlxWhisperBackend(WhisperBackend):
    """MLX backend adapter for WhisperX - corrected version."""
    
    def __init__(
        self,
        model: str,
        device: str = "mlx",
        compute_type: str = "float16",
        options: Optional[Dict] = None,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        batch_size: int = 8,
        word_timestamps: bool = False,
        **kwargs
    ):
        """Initialize MLX Whisper backend.
        
        Args:
            model: Model name or path
            device: Device (always "mlx")
            compute_type: Precision ("float16", "float32", "int4")
            options: Decoding options
            download_root: Download directory (unused - MLX handles this)
            local_files_only: Whether to use only local files
            batch_size: Batch size for processing
            word_timestamps: Whether to extract word timestamps
        """
        # No need to call super().__init__ since it's abstract
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.options = options or {}
        
        self.batch_size = batch_size
        self.word_timestamps = word_timestamps
        self.kwargs = kwargs
        
        # Model name mapping for MLX
        self.model_name_map = {
            # Standard models
            "tiny": "mlx-community/whisper-tiny",
            "tiny.en": "mlx-community/whisper-tiny.en",
            "base": "mlx-community/whisper-base",
            "base.en": "mlx-community/whisper-base.en",
            "small": "mlx-community/whisper-small",
            "small.en": "mlx-community/whisper-small.en",
            "medium": "mlx-community/whisper-medium",
            "medium.en": "mlx-community/whisper-medium.en",
            "large": "mlx-community/whisper-large",
            "large-v1": "mlx-community/whisper-large-v1",
            "large-v2": "mlx-community/whisper-large-v2",
            "large-v3": "mlx-community/whisper-large-v3",
            # INT4 quantized models
            "tiny-q4": "mlx-community/whisper-tiny-mlx-q4",
            "base-q4": "mlx-community/whisper-base-mlx-q4",
            "small-q4": "mlx-community/whisper-small-mlx-q4",
            "medium-q4": "mlx-community/whisper-medium-mlx-q4",
            "large-q4": "mlx-community/whisper-large-mlx-q4",
            "large-v3-q4": "mlx-community/whisper-large-v3-mlx-q4",
        }
        
        # Map model name if needed
        if model in self.model_name_map:
            self.path_or_hf_repo = self.model_name_map[model]
        else:
            self.path_or_hf_repo = model
        
        # For lazy initialization
        self.model = None
        self._model_holder = None
    
    def _lazy_init(self):
        """Lazy initialization as per roadmap pattern."""
        if self.model is None:
            # Use ModelHolder pattern from mlx_whisper for caching
            from mlx_whisper.transcribe import ModelHolder
            self._model_holder = ModelHolder
            dtype = mx.float16 if self.compute_type == "float16" else mx.float32
            self.model = self._model_holder.get_model(self.path_or_hf_repo, dtype)
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using MLX Whisper.
        
        Uses mlx_whisper.transcribe() directly for auto-download support.
        """
        # Disable numba JIT to avoid OpenMP conflicts with PyTorch
        import os
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        
        # Import MLX whisper
        import mlx_whisper
        
        # Ensure audio is loaded
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Prepare decode options
        decode_options = {
            "path_or_hf_repo": self.path_or_hf_repo,
            "task": task,
            "language": language,
            "word_timestamps": self.word_timestamps or kwargs.get("word_timestamps", False),
            "verbose": print_progress,
        }
        
        # Handle temperature parameter correctly
        if "temperatures" in self.options:
            temps = self.options["temperatures"]
            if isinstance(temps, (list, tuple)):
                # mlx_whisper expects tuple for multiple temperatures
                decode_options["temperature"] = temps
            else:
                decode_options["temperature"] = temps
        else:
            # Default to greedy decoding
            decode_options["temperature"] = 0.0
        
        # Map options from WhisperX format to MLX format
        option_mapping = {
            "beam_size": "beam_size",
            "best_of": "best_of",
            "patience": "patience",
            "length_penalty": "length_penalty",
            "initial_prompt": "initial_prompt",
            "suppress_tokens": "suppress_tokens",
            "suppress_blank": "suppress_blank",
            "log_prob_threshold": "logprob_threshold",
            "no_speech_threshold": "no_speech_threshold",
            "compression_ratio_threshold": "compression_ratio_threshold",
            "condition_on_previous_text": "condition_on_previous_text",
            "prepend_punctuations": "prepend_punctuations",
            "append_punctuations": "append_punctuations",
        }
        
        for wx_key, mlx_key in option_mapping.items():
            if wx_key in self.options:
                decode_options[mlx_key] = self.options[wx_key]
        
        # Add any additional kwargs
        decode_options.update(kwargs)
        
        # Remove unsupported options
        for key in ["vad_filter", "vad_parameters", "max_new_tokens", "batch_size", "num_workers"]:
            decode_options.pop(key, None)
        
        # Transcribe using MLX (auto-downloads model if needed)
        result = mlx_whisper.transcribe(
            audio,
            **decode_options
        )
        
        # Convert result to WhisperX format
        segments = []
        for segment in result.get("segments", []):
            seg = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            
            # Add word timestamps if available
            if "words" in segment:
                seg["words"] = segment["words"]
            
            segments.append(seg)
        
        return {
            "segments": segments,
            "language": result.get("language", language or "en")
        }
    
    def transcribe_batch(
        self,
        audio_segments: List[np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio segments.
        
        Future: When MLX supports batching, update this method.
        """
        results = []
        
        for i, audio in enumerate(audio_segments):
            if kwargs.get("print_progress", False):
                print(f"Processing segment {i+1}/{len(audio_segments)}")
            
            result = self.transcribe(
                audio,
                language=language,
                task=task,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def detect_language(self, audio: Union[str, np.ndarray]) -> str:
        """Detect language of audio."""
        # Disable numba JIT to avoid OpenMP conflicts
        import os
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        
        import mlx_whisper
        
        # Ensure audio is loaded
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Use first 30 seconds for detection
        if len(audio) > 30 * 16000:
            audio = audio[:30 * 16000]
        
        # Detect language
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.path_or_hf_repo,
            task="transcribe",
            verbose=False,
            temperature=0.0,
        )
        
        return result.get("language", "en")
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        # List from OpenAI Whisper
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        ]
    
    @property
    def is_multilingual(self) -> bool:
        """Return whether the model supports multiple languages."""
        # Check if it's an English-only model
        return not (self.path_or_hf_repo.endswith(".en") or ".en-" in self.path_or_hf_repo)


class MlxWhisper:
    """MLX Whisper adapter matching roadmap API specification."""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        dtype: str = "float16",
        word_timestamps: bool = False,
        language: Optional[str] = None,
        **opts
    ):
        """Initialize MlxWhisper adapter.
        
        Args:
            model_path: Path to model or HF repo (e.g., "~/mlx_models/large-v3-int4")
            batch_size: Batch size for processing
            dtype: Model precision ("float16", "float32", "int4")
            word_timestamps: Extract word timestamps
            language: Language code
            **opts: Additional options
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.dtype = dtype
        self.word_timestamps = word_timestamps
        self.language = language
        self.opts = opts
        self.model = None
    
    def _lazy_init(self):
        """Lazy model initialization as per roadmap."""
        if self.model is None:
            import mlx_whisper
            from mlx_whisper.load_models import load_model
            
            # Determine MLX dtype
            if self.dtype == "int4":
                # INT4 models are pre-quantized
                mlx_dtype = mx.float16
            else:
                mlx_dtype = mx.float16 if self.dtype == "float16" else mx.float32
            
            # Expand path
            import os
            model_path = os.path.expanduser(self.model_path)
            
            # Load model
            self.model = load_model(model_path, dtype=mlx_dtype)
    
    def transcribe(self, wav_path: str) -> Dict:
        """Transcribe audio file matching roadmap API.
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Transcription result with segments and metadata
        """
        self._lazy_init()
        
        # Load audio using WhisperX helper
        from whisperx.audio import load_audio
        audio = load_audio(wav_path, sr=16_000)
        
        # Disable numba JIT to avoid OpenMP conflicts
        import os
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        
        # Transcribe
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio,
            model=self.model,
            batch_size=self.batch_size,
            word_timestamps=self.word_timestamps,
            language=self.language,
            **self.opts
        )
        
        return result