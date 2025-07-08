#!/usr/bin/env python3
"""
MLX implementation of Voice Activity Detection
Replaces PyTorch-based VAD with pure MLX implementation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from whisperx.audio import SAMPLE_RATE


class MLXSileroVAD(nn.Module):
    """MLX implementation of Silero VAD."""
    
    def __init__(self, model_path: str):
        """Initialize VAD model.
        
        Args:
            model_path: Path to MLX model directory
        """
        super().__init__()
        
        # Load config
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Model parameters
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.window_size_samples = self.config.get("window_size_samples", 512)
        
        # Build model
        self.lstm1 = nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=self.config["hidden_size"],
            hidden_size=self.config["hidden_size"],
            batch_first=True
        )
        
        self.output = nn.Linear(
            self.config["hidden_size"],
            self.config["num_classes"]
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Load weights
        weights_path = Path(model_path) / "weights.npz"
        weights = mx.load(str(weights_path))
        self.load_weights(list(weights.items()))
        
        # Thresholds
        self.threshold = 0.5
        self.min_speech_duration_ms = 250
        self.min_silence_duration_ms = 100
        self.window_size_ms = 32  # 512 samples at 16kHz
        self.speech_pad_ms = 30
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
    
    def __call__(
        self,
        audio: Union[np.ndarray, mx.array],
        threshold: Optional[float] = None,
        min_speech_duration_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Process audio and return speech segments.
        
        Args:
            audio: Audio data (numpy or MLX array)
            threshold: Speech probability threshold
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration
            
        Returns:
            List of segments with start/end times in seconds
        """
        if threshold is not None:
            self.threshold = threshold
        if min_speech_duration_ms is not None:
            self.min_speech_duration_ms = min_speech_duration_ms
        if min_silence_duration_ms is not None:
            self.min_silence_duration_ms = min_silence_duration_ms
        
        # Convert to MLX array if needed
        if isinstance(audio, np.ndarray):
            audio = mx.array(audio)
        
        # Process audio in windows
        probabilities = self._get_speech_probabilities(audio)
        
        # Convert to segments
        segments = self._probabilities_to_segments(probabilities)
        
        return segments
    
    def _get_speech_probabilities(self, audio: mx.array) -> mx.array:
        """Get speech probabilities for audio windows.
        
        Args:
            audio: Audio data
            
        Returns:
            Speech probabilities for each window
        """
        # Ensure correct sample rate
        if len(audio.shape) == 1:
            audio = audio[None, :]  # Add batch dimension
        
        # Sliding window approach
        num_samples = audio.shape[1]
        num_windows = (num_samples - self.window_size_samples) // self.window_size_samples + 1
        
        probabilities = []
        
        for i in range(num_windows):
            start = i * self.window_size_samples
            end = start + self.window_size_samples
            
            # Extract window
            window = audio[:, start:end]
            
            # Normalize audio
            window = window / 32768.0  # Assuming 16-bit audio
            
            # Add feature dimension
            window = window[:, :, None]  # Shape: (batch, time, features)
            
            # Get probability
            prob = self.forward(window)
            probabilities.append(prob)
        
        # Stack probabilities
        probabilities = mx.concatenate(probabilities, axis=1)
        
        return probabilities[0]  # Remove batch dimension
    
    def _probabilities_to_segments(
        self,
        probabilities: mx.array
    ) -> List[Dict[str, float]]:
        """Convert speech probabilities to segments.
        
        Args:
            probabilities: Speech probabilities
            
        Returns:
            List of speech segments
        """
        # Convert to numpy for easier processing
        probs = np.array(probabilities)
        
        # Apply threshold
        speech = probs > self.threshold
        
        # Find speech regions
        segments = []
        in_speech = False
        start_idx = 0
        
        # Convert durations to window counts
        min_speech_windows = int(self.min_speech_duration_ms / self.window_size_ms)
        min_silence_windows = int(self.min_silence_duration_ms / self.window_size_ms)
        pad_windows = int(self.speech_pad_ms / self.window_size_ms)
        
        for i, is_speech in enumerate(speech):
            if is_speech and not in_speech:
                # Start of speech
                start_idx = max(0, i - pad_windows)
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech
                end_idx = min(len(speech), i + pad_windows)
                
                # Check minimum duration
                if end_idx - start_idx >= min_speech_windows:
                    segments.append({
                        "start": start_idx * self.window_size_samples / self.sample_rate,
                        "end": end_idx * self.window_size_samples / self.sample_rate
                    })
                
                in_speech = False
        
        # Handle final segment
        if in_speech:
            end_idx = len(speech)
            if end_idx - start_idx >= min_speech_windows:
                segments.append({
                    "start": start_idx * self.window_size_samples / self.sample_rate,
                    "end": end_idx * self.window_size_samples / self.sample_rate
                })
        
        # Merge close segments
        segments = self._merge_close_segments(segments, min_silence_windows)
        
        return segments
    
    def _merge_close_segments(
        self,
        segments: List[Dict[str, float]],
        min_silence_windows: int
    ) -> List[Dict[str, float]]:
        """Merge segments that are close together.
        
        Args:
            segments: List of segments
            min_silence_windows: Minimum silence duration in windows
            
        Returns:
            Merged segments
        """
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        min_silence_seconds = min_silence_windows * self.window_size_samples / self.sample_rate
        
        for segment in segments[1:]:
            last = merged[-1]
            gap = segment["start"] - last["end"]
            
            if gap < min_silence_seconds:
                # Merge segments
                last["end"] = segment["end"]
            else:
                # Keep separate
                merged.append(segment)
        
        return merged


class MLXVAD:
    """High-level VAD interface compatible with WhisperX."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "mlx",
    ):
        """Initialize VAD.
        
        Args:
            model_path: Path to MLX VAD model
            device: Device (always "mlx")
        """
        if model_path is None:
            model_path = self._get_default_model_path()
        
        self.model = MLXSileroVAD(model_path)
        self.device = device
    
    def _get_default_model_path(self) -> str:
        """Get default model path."""
        # Check common locations
        paths = [
            Path.home() / "mlx_models" / "vad" / "silero_vad_mlx",
            Path("mlx_models") / "vad" / "silero_vad_mlx",
            Path("/tmp/mlx_models/vad/silero_vad_mlx"),
        ]
        
        for path in paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "No MLX VAD model found. Please convert Silero VAD using:\n"
            "python -m whisperx.convert_vad_models --download --output ~/mlx_models/vad/silero_vad_mlx"
        )
    
    def __call__(
        self,
        audio: Union[str, np.ndarray],
        **kwargs
    ) -> Tuple[List[Dict[str, float]], Dict]:
        """Process audio and return segments.
        
        Args:
            audio: Audio file path or numpy array
            **kwargs: Additional arguments
            
        Returns:
            Segments and info dict
        """
        # Load audio if path
        if isinstance(audio, str):
            from whisperx.audio import load_audio
            audio = load_audio(audio)
        
        # Get segments
        segments = self.model(audio, **kwargs)
        
        # Info dict for compatibility
        info = {
            "language": None,
            "language_probability": None,
            "duration": len(audio) / SAMPLE_RATE,
        }
        
        return segments, info


def load_vad_model(
    model_name: str = "silero",
    device: str = "mlx",
    model_path: Optional[str] = None,
    **kwargs
) -> MLXVAD:
    """Load VAD model.
    
    Args:
        model_name: Model name (only "silero" supported)
        device: Device (always "mlx")
        model_path: Optional path to MLX model
        **kwargs: Additional arguments
        
    Returns:
        VAD model instance
    """
    if model_name != "silero":
        raise ValueError(f"Only 'silero' VAD is supported, got {model_name}")
    
    return MLXVAD(model_path=model_path, device=device)


# Compatibility function
def load_vad_model_mlx(
    model_path: Optional[str] = None,
    device: str = "mlx"
) -> MLXVAD:
    """Load MLX VAD model (compatibility wrapper)."""
    return load_vad_model("silero", device, model_path)