"""
WhisperX Backends - MLX Only

This MLX-only fork of WhisperX exclusively uses the MLX backend for Apple Silicon.
"""
from .mlx_whisper import MlxWhisperBackend

__all__ = ["MlxWhisperBackend"]