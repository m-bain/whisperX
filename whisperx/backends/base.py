from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from whisperx.types import TranscriptionResult


class WhisperBackend(ABC):
    """Abstract base class for Whisper backends."""
    
    @abstractmethod
    def __init__(
        self,
        model: str,
        device: str,
        device_index: int = 0,
        compute_type: str = "float16",
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        threads: int = 4,
        **kwargs
    ):
        """Initialize the backend with model parameters."""
        pass
    
    @abstractmethod
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
        """Transcribe audio using the backend."""
        pass
    
    @abstractmethod
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio."""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        pass
    
    @property
    @abstractmethod
    def is_multilingual(self) -> bool:
        """Return whether the model supports multiple languages."""
        pass