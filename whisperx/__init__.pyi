# whisperx/__init__.pyi

# This file defines the static interface for the package. It is read by
# type checkers but is completely ignored by the Python interpreter at runtime.

# Re-export the public functions directly from their submodules to provide
# their exact signatures to the type checker.
from .alignment import align, load_align_model
from .asr import load_model
from .audio import load_audio
from .diarize import (
    assign_word_speakers,
    DiarizationPipeline,
)

# Also re-export any types used in the function signatures so the type
# checker can resolve them (e.g., AlignedTranscriptionResult).
from .types import (  # noqa: F401
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SingleCharSegment,
    TranscriptionResult,
)

__all__ = [
    "load_align_model",
    "align",
    "load_model",
    "load_audio",
    "assign_word_speakers",
    "DiarizationPipeline",
]
