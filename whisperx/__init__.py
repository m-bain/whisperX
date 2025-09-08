from typing import Iterable, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch

    from whisperx.types import (
        AlignedTranscriptionResult,
        TranscriptionResult,
        SingleSegment,
    )

    from whisperx.diarize import DiarizationPipeline


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir=None
) -> Tuple["torch.nn.Module", dict]:
    """Load an alignment model for forced alignment.
    
    Returns:
        Tuple of (align_model, align_metadata)
    """
    import whisperx.alignment as alignment
    return alignment.load_align_model(
        language_code, device, model_name, model_dir
    )


def align(
    transcript: Iterable["SingleSegment"],
    model: "torch.nn.Module",
    align_model_metadata: dict,
    audio: Union[str, "np.ndarray", "torch.Tensor"],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> "AlignedTranscriptionResult":
    """Align phoneme recognition predictions to known transcription."""
    import whisperx.alignment as alignment
    return alignment.align(
        transcript, model, align_model_metadata, audio, device,
        interpolate_method, return_char_alignments, print_progress,
        combined_progress
    )


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model=None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model=None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
):
    """Load a Whisper model for inference."""
    import whisperx.asr as asr
    return asr.load_model(
        whisper_arch, device, device_index, compute_type, asr_options,
        language, vad_model, vad_method, vad_options, model, task,
        download_root, local_files_only, threads
    )


def load_audio(file: str, sr: int = 16000) -> "np.ndarray":
    """Open an audio file and read as mono waveform, resampling as necessary.
    
    Returns:
        A NumPy array containing the audio waveform, in float32 dtype.
    """
    import whisperx.audio as audio
    return audio.load_audio(file, sr)


def load_diarization_pipeline(
    model_name=None,
    use_auth_token=None,
    device: Optional[Union[str, "torch.device"]] = "cpu",
 ) -> "DiarizationPipeline":
    """Create a speaker diarization pipeline.
    
    Args:
        model_name: Name of the diarization model to use
        use_auth_token: HuggingFace authentication token if required
        device: Device to run the model on ('cpu', 'cuda', or torch.device)
    
    Returns:
        DiarizationPipeline instance
    """
    import whisperx.diarize as diarize
    return diarize.DiarizationPipeline(
        model_name=model_name,
        use_auth_token=use_auth_token,
        device=device
    )


def assign_word_speakers(
    diarize_df: "pd.DataFrame",
    transcript_result: Union[
        "AlignedTranscriptionResult", "TranscriptionResult"
    ],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union["AlignedTranscriptionResult", "TranscriptionResult"]:
    """Assign speakers to words and segments in the transcript."""
    import whisperx.diarize as diarize
    return diarize.assign_word_speakers(
        diarize_df, transcript_result, speaker_embeddings, fill_nearest
    )
