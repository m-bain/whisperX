# This file is only run by type checkers, not at runtime. It defines the static
# interface for the package for easier type checking.
from .alignment import align, load_align_model
from .asr import load_model, FasterWhisperPipeline
from .audio import load_audio
from .diarize import (
    assign_word_speakers,
    DiarizationPipeline,
)

# Also re-export any types used in the function signatures so the type
# checker can resolve them
from .types import (
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
    "FasterWhisperPipeline",
    "AlignedTranscriptionResult",
    "SingleSegment",
    "SingleAlignedSegment",
    "SingleWordSegment",
    "SingleCharSegment",
    "TranscriptionResult",
]
