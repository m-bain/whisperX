import os
from typing import Optional
import warnings

from whisperx.backends import FasterWhisperBackend, MlxWhisperBackend
from whisperx.backends.base import WhisperBackend


def load_model(
    whisper_arch: str,
    backend: str = "faster-whisper",
    device: str = "cuda",
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_method: str = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
) -> WhisperBackend:
    """Load a Whisper model for inference with the specified backend.
    
    Args:
        whisper_arch: The name of the Whisper model to load.
        backend: The backend to use ("faster-whisper" or "mlx").
        device: The device to load the model on.
        device_index: The device index to use.
        compute_type: The compute type to use for the model.
        asr_options: ASR options dictionary.
        language: The language of the model.
        vad_method: The VAD method to use.
        vad_options: VAD options dictionary.
        task: The task to perform ("transcribe" or "translate").
        download_root: The root directory to download the model to.
        local_files_only: If True, avoid downloading the file and return the path to the local cached file if it exists.
        threads: The number of CPU threads to use per worker.
        
    Returns:
        A WhisperBackend instance.
    """
    
    # Select backend implementation
    if backend == "faster-whisper":
        return FasterWhisperBackend(
            model=whisper_arch,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
            threads=threads,
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options=vad_options,
            language=language,
            task=task,
        )
    elif backend == "mlx":
        if MlxWhisperBackend is None:
            raise ImportError(
                "MLX backend is not available. Please install mlx-whisper: "
                "pip install mlx-whisper"
            )
        
        # Check if running on Apple Silicon
        import platform
        if platform.machine() not in ["arm64", "aarch64"]:
            warnings.warn(
                "MLX backend is optimized for Apple Silicon. "
                "Performance may be suboptimal on other platforms."
            )
        
        # For MLX backend, we need to handle model paths differently
        # MLX expects either a local path or HuggingFace repo ID
        if "/" not in whisper_arch and not os.path.exists(whisper_arch):
            # Convert model size to HF repo format
            model_map = {
                "tiny": "mlx-community/whisper-tiny",
                "tiny.en": "mlx-community/whisper-tiny.en",
                "base": "mlx-community/whisper-base", 
                "base.en": "mlx-community/whisper-base.en",
                "small": "mlx-community/whisper-small",
                "small.en": "mlx-community/whisper-small.en",
                "medium": "mlx-community/whisper-medium",
                "medium.en": "mlx-community/whisper-medium.en",
                "large": "mlx-community/whisper-large",
                "large-v2": "mlx-community/whisper-large-v2",
                "large-v3": "mlx-community/whisper-large-v3",
            }
            
            if whisper_arch in model_map:
                mlx_model_path = model_map[whisper_arch]
            else:
                # Try to use the model name as-is
                mlx_model_path = whisper_arch
        else:
            mlx_model_path = whisper_arch
        
        return MlxWhisperBackend(
            model=mlx_model_path,
            device="mlx",  # MLX always uses Apple Silicon
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
            threads=threads,
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options=vad_options,
            language=language,
            task=task,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# For backwards compatibility, keep the original imports
# These are used by alignment.py and other modules
from whisperx.backends.faster_whisper import (
    WhisperModel,
    FasterWhisperPipeline,
    find_numeral_symbol_tokens
)