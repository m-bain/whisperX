# whisperx/__init__.py

import importlib
from typing import Any

# This list defines the public API. It is the single source of truth
# for what this package will lazily export at runtime.

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


def __getattr__(name: str) -> Any:
    """
    Lazily import functions when they are first accessed, based on PEP 562.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Map the requested function to its source submodule.
    if name in ("load_align_model", "align"):
        module_path = ".alignment"
    elif name in ("FasterWhisperPipeline", "load_model"):
        module_path = ".asr"
    elif name == "load_audio":
        module_path = ".audio"
    elif name in (
        "assign_word_speakers",
        "DiarizationPipeline",
    ):
        module_path = ".diarize"
    elif name in (
        "AlignedTranscriptionResult",
        "SingleSegment",
        "SingleAlignedSegment",
        "SingleWordSegment",
        "SingleCharSegment",
        "TranscriptionResult",
    ):
        module_path = ".types"
    else:
        # This case should not be reached if __all__ is accurate.
        raise ImportError(f"Cannot determine the source for {name}")

    # Perform the actual import of the submodule.
    module = importlib.import_module(module_path, __name__)

    # Get the function from the imported module.
    attribute = getattr(module, name)

    # Cache the attribute in this module's globals. This is a crucial
    # optimization that ensures this function is not called again for the
    # same name.
    globals()[name] = attribute

    return attribute


def __dir__() -> list[str]:
    """
    Expose the lazily loaded attributes for dir() and autocompletion tools.
    """
    return list(globals().keys()) + __all__
