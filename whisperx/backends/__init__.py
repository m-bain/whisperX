from .faster_whisper import FasterWhisperBackend

# Conditional import for MLX backend (only available on Apple Silicon)
try:
    from .mlx_whisper import MlxWhisperBackend
    __all__ = ["FasterWhisperBackend", "MlxWhisperBackend"]
except ImportError:
    __all__ = ["FasterWhisperBackend"]
    MlxWhisperBackend = None