import os
from typing import List, Optional, Union
from dataclasses import replace
import warnings

import numpy as np
import torch
import mlx.core as mx
import mlx_whisper

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.types import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote
from .base import WhisperBackend


class MlxWhisperBackend(WhisperBackend):
    """Backend implementation for mlx-whisper on Apple Silicon."""
    
    def __init__(
        self,
        model: str,
        device: str = "mlx",  # MLX always runs on Apple Silicon
        device_index: int = 0,  # Not used for MLX
        compute_type: str = "float16",
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        threads: int = 4,  # Not used for MLX
        asr_options: Optional[dict] = None,
        vad_method: str = "pyannote",
        vad_options: Optional[dict] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        batch_size: int = 8,
        **kwargs
    ):
        self.model_path = model
        self.batch_size = batch_size
        self.dtype = compute_type
        self.language = language
        self.task = task
        
        # Convert compute_type to MLX dtype
        self.mlx_dtype = mx.float16 if compute_type == "float16" else mx.float32
        
        # Load model immediately instead of lazy loading to avoid issues
        from mlx_whisper.load_models import load_model
        self.model = load_model(self.model_path, dtype=self.mlx_dtype)
        self._model_loaded = True
        
        # Setup ASR options - separate transcribe options from decoding options
        self.default_asr_options = {
            # Options for mlx_whisper.transcribe
            "temperature": 0.01,  # Use small temperature > 0 to avoid beam search
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "word_timestamps": True,
            "prepend_punctuations": "\"'\u2018\u00BF([{-",
            "append_punctuations": "\"'.。,，!！?？:：\")]}、",
            "hallucination_silence_threshold": None,
            # DecodingOptions parameters
            "language": language,
            "task": task,
            "fp16": compute_type == "float16",
        }
        
        if asr_options is not None:
            self.default_asr_options.update(asr_options)
        
        # Setup VAD
        default_vad_options = {
            "chunk_size": 30,
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        
        if vad_options is not None:
            default_vad_options.update(vad_options)
            
        self.vad_params = default_vad_options
        
        # Initialize VAD model
        if vad_method == "silero":
            self.vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            # VAD runs on CPU for now
            vad_device = torch.device("cpu")
            self.vad_model = Pyannote(vad_device, use_auth_token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")
    
    def _lazy_init(self):
        """Lazy initialization of the MLX model."""
        # Model is now loaded in __init__, so this is a no-op
        pass
    
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
        verbose: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using MLX Whisper backend."""
        self._lazy_init()
        
        # Load audio if path is provided
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Pre-process audio and merge chunks using VAD
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
        
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size or self.vad_params["chunk_size"],
            onset=self.vad_params["vad_onset"],
            offset=self.vad_params["vad_offset"],
        )
        
        # Update language and task if provided
        if language is not None:
            self.default_asr_options["language"] = language
        elif self.language is not None:
            self.default_asr_options["language"] = self.language
            
        if task is not None:
            self.default_asr_options["task"] = task
        elif self.task is not None:
            self.default_asr_options["task"] = self.task
        
        segments: List[SingleSegment] = []
        total_segments = len(vad_segments)
        
        # Process each VAD segment
        for idx, vad_segment in enumerate(vad_segments):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            
            # Extract audio segment
            f1 = int(vad_segment['start'] * SAMPLE_RATE)
            f2 = int(vad_segment['end'] * SAMPLE_RATE)
            audio_segment = audio[f1:f2]
            
            # Transcribe using MLX
            transcribe_options = self.default_asr_options.copy()
            transcribe_options.update(kwargs)
            
            # Remove verbose from options if it exists, pass it separately
            transcribe_options.pop('verbose', None)
            
            # Convert 'temperatures' to 'temperature' (MLX expects singular)
            if 'temperatures' in transcribe_options:
                temps = transcribe_options.pop('temperatures')
                if isinstance(temps, (list, tuple)) and len(temps) > 0:
                    transcribe_options['temperature'] = temps[0]
                else:
                    transcribe_options['temperature'] = temps
                    
            # Convert log_prob_threshold to logprob_threshold (MLX uses no underscore)
            if 'log_prob_threshold' in transcribe_options:
                transcribe_options['logprob_threshold'] = transcribe_options.pop('log_prob_threshold')
                    
            # MLX doesn't support beam search yet, so remove beam_size when temperature is 0
            if transcribe_options.get('temperature', 0.0) == 0.0:
                transcribe_options.pop('beam_size', None)
                transcribe_options.pop('patience', None)
                transcribe_options.pop('best_of', None)
                
            # Remove unsupported options
            for unsupported in ['suppress_numerals', 'max_new_tokens', 'clip_timestamps', 
                              'repetition_penalty', 'no_repeat_ngram_size', 
                              'prompt_reset_on_temperature', 'prefix', 'suppress_blank',
                              'suppress_tokens', 'without_timestamps', 'max_initial_timestamp',
                              'multilingual', 'hotwords', 'batch_size', 'num_workers']:
                transcribe_options.pop(unsupported, None)
            
            result = mlx_whisper.transcribe(
                audio_segment,
                path_or_hf_repo=self.model_path,
                verbose=verbose,
                **transcribe_options
            )
            
            # Extract text from result
            text = result.get("text", "").strip()
            
            if verbose:
                print(f"Transcript: [{round(vad_segment['start'], 3)} --> {round(vad_segment['end'], 3)}] {text}")
            
            # Add segment with proper timestamps
            segments.append({
                "text": text,
                "start": round(vad_segment['start'], 3),
                "end": round(vad_segment['end'], 3)
            })
        
        # Get language from transcription result if not already set
        detected_language = self.default_asr_options.get("language", "en")
        
        return {"segments": segments, "language": detected_language}
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio using MLX."""
        self._lazy_init()
        
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        
        # Use MLX transcribe with language detection
        result = mlx_whisper.transcribe(
            audio[:N_SAMPLES],  # Use first 30 seconds
            path_or_hf_repo=self.model_path,
            verbose=False,
            language=None,  # Auto-detect language
            fp16=self.mlx_dtype == mx.float16
        )
        
        detected_language = result.get("language", "en")
        print(f"Detected language: {detected_language}")
        return detected_language
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        # MLX Whisper supports all Whisper languages
        return list(mlx_whisper.tokenizer.LANGUAGES.keys())
    
    @property  
    def is_multilingual(self) -> bool:
        """Return whether the model supports multiple languages."""
        # Check model name to determine if it's multilingual
        model_name = self.model_path.lower()
        return not model_name.endswith('.en')