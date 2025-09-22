"""
Configuration management for WhisperX desktop application.
Handles user preferences, model settings, and preset management.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class TranscriptionConfig:
    """Configuration parameters for WhisperX transcription."""

    # File settings
    audio_file: str = ""

    # Model settings
    model_name: str = "large-v2"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"

    # Processing options
    enable_alignment: bool = True
    enable_diarization: bool = False
    language: Optional[str] = None

    # Advanced settings
    batch_size: int = 8
    # chunk_size: int = 30
    # vad_method: str = "pyannote"
    # vad_onset: float = 0.500
    # vad_offset: float = 0.363

    # Output settings
    show_timestamps: bool = False
    show_speakers: bool = False

    def to_whisperx_params(self) -> Dict[str, Any]:
        """Convert config to WhisperX transcription parameters."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'device_index': self.device_index,
            'compute_type': self.compute_type,
            'batch_size': self.batch_size,
            # 'chunk_size': self.chunk_size,
            'language': self.language,
            # 'vad_method': self.vad_method,
            # 'vad_onset': self.vad_onset,
            # 'vad_offset': self.vad_offset,
            'diarize': self.enable_diarization,
            'no_align': not self.enable_alignment,
            'print_progress': True,
        }


class AppConfig:
    """Manages application configuration and user presets."""

    def __init__(self):
        self.config_dir = Path.home() / '.whisperx_app'
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / 'config.json'
        self.presets_file = self.config_dir / 'presets.json'

        self._current_config = TranscriptionConfig()
        self._presets: Dict[str, Dict[str, Any]] = {}

        self.load_config()
        self.load_presets()

    def __str__(self):
        return str(self.config_dir) + " | " + str(self.config_file) + " | " + str(self.presets_file) + " | " +  str(self._presets) + " |\ "

    def get_current_config(self) -> TranscriptionConfig:
        """Get current transcription configuration."""
        return self._current_config

    def update_config(self, **kwargs) -> None:
        """Update current configuration with new values."""
        print("Updating config")
        for key, value in kwargs.items():
            print("KEY VAL", key, " ", value)
            if hasattr(self._current_config, key):
                print("HAS KEY VAL", key, " ", value)
                setattr(self._current_config, key, value)
        self.save_config()

    def load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._current_config = TranscriptionConfig(**data)
            except Exception as e:
                print(f"Error loading config: {e}")
                self._current_config = TranscriptionConfig()

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self._current_config), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def save_preset(self, name: str) -> None:
        """Save current config as a named preset."""
        self._presets[name] = asdict(self._current_config)
        self._save_presets()

    def load_preset(self, name: str) -> bool:
        """Load a named preset."""
        if name in self._presets:
            try:
                self._current_config = TranscriptionConfig(**self._presets[name])
                self.save_config()
                return True
            except Exception as e:
                print(f"Error loading preset {name}: {e}")
        return False

    def get_preset_names(self) -> list:
        """Get list of available preset names."""
        return list(self._presets.keys())

    def delete_preset(self, name: str) -> bool:
        """Delete a named preset."""
        if name in self._presets:
            del self._presets[name]
            self._save_presets()
            return True
        return False

    def load_presets(self) -> None:
        """Load presets from file."""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r') as f:
                    self._presets = json.load(f)
            except Exception as e:
                print(f"Error loading presets: {e}")
                self._presets = {}

    def _save_presets(self) -> None:
        """Save presets to file."""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self._presets, f, indent=2)
        except Exception as e:
            print(f"Error saving presets: {e}")