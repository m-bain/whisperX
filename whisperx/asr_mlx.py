"""Apple Silicon (MLX) ASR backend.

A drop-in alternative to the faster-whisper path in :mod:`whisperx.asr`: same
``load_model`` entry contract (reached via ``device="mlx"`` there) and the same
``transcribe`` return shape (``{"segments", "language"}``), so the downstream
align/diarize stages are untouched.

Adapted from the KalebJS/whispermlx fork (https://github.com/KalebJS/whispermlx);
``MLXWhisperPipeline``, ``MLX_MODEL_MAP`` and the per-chunk transcribe loop are
lifted from its ``whispermlx/asr.py`` with minor changes (lazy mlx import, our
package paths). ``mlx_whisper`` (Apple Silicon only) is imported lazily so
``import whisperx`` stays cheap and cross-platform — nothing here touches mlx
until a ``device="mlx"`` model is actually loaded/run.
"""

from __future__ import annotations

import contextlib
import io
from typing import Optional, Union

import numpy as np

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback, SingleSegment, TranscriptionResult
from whisperx.vads import Pyannote, Silero, Vad

logger = get_logger(__name__)

# Short Whisper names -> mlx-community HF repos (separate from the CT2 checkpoints).
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-mlx",
    "large-v1": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def _resolve_mlx_model(whisper_arch: str) -> str:
    """Map a short name or full HF repo ID to an mlx-community repo."""
    if "/" in whisper_arch:
        return whisper_arch
    if whisper_arch in MLX_MODEL_MAP:
        return MLX_MODEL_MAP[whisper_arch]
    logger.warning(
        "Unknown model '%s'. Pass a full HF repo ID or one of: %s",
        whisper_arch, list(MLX_MODEL_MAP.keys()),
    )
    return whisper_arch


def _compute_avg_logprob(mlx_segments: list) -> float:
    if not mlx_segments:
        return 0.0
    total, weighted = 0, 0.0
    for seg in mlx_segments:
        n = len(seg.get("tokens", [])) or 1
        weighted += seg.get("avg_logprob", 0.0) * n
        total += n
    return weighted / total if total else 0.0


class MLXWhisperPipeline:
    """WhisperX transcription pipeline using mlx-whisper on Apple Silicon.

    Mirrors the public surface of :class:`whisperx.asr.FasterWhisperPipeline`
    that the rest of the pipeline relies on: a ``transcribe`` method returning a
    :class:`~whisperx.schema.TranscriptionResult`. There is no batching — VAD
    segments are transcribed serially; MLX uses the GPU automatically.
    """

    def __init__(
        self,
        model_path: str,
        vad,
        vad_params: dict,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ):
        self.model_path = model_path
        self.vad_model = vad
        self._vad_params = vad_params
        self.preset_language = language
        self.task = task
        self.initial_prompt = initial_prompt

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
        progress_callback: ProgressCallback = None,
    ) -> TranscriptionResult:
        import mlx_whisper  # lazy: Apple-Silicon-only dependency
        from tqdm import tqdm

        if isinstance(audio, str):
            audio = load_audio(audio)

        effective_language = language or self.preset_language
        effective_task = task or self.task

        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: list[SingleSegment] = []
        total_segments = len(vad_segments)

        pbar = tqdm(total=total_segments, desc="Transcribing", unit="seg")
        for idx, vad_seg in enumerate(vad_segments):
            f1 = int(vad_seg["start"] * SAMPLE_RATE)
            f2 = int(vad_seg["end"] * SAMPLE_RATE)
            audio_chunk = audio[f1:f2]

            with contextlib.redirect_stderr(io.StringIO()):
                mlx_result = mlx_whisper.transcribe(
                    audio_chunk,
                    path_or_hf_repo=self.model_path,
                    language=effective_language,
                    task=effective_task,
                    verbose=False,
                    initial_prompt=self.initial_prompt,
                    word_timestamps=False,  # word timing comes from the align stage
                )

            if effective_language is None and idx == 0:
                effective_language = mlx_result.get("language")

            chunk_text = mlx_result.get("text", "").strip()
            avg_logprob = _compute_avg_logprob(mlx_result.get("segments", []))

            pbar.update(1)
            if verbose:
                tqdm.write(
                    f"[{round(vad_seg['start'], 3)} --> {round(vad_seg['end'], 3)}] {chunk_text}"
                )
            if print_progress:
                base = ((idx + 1) / total_segments) * 100
                pct = base / 2 if combined_progress else base
                tqdm.write(f"Progress: {pct:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

            segments.append(
                {
                    "text": chunk_text,
                    "start": round(vad_seg["start"], 3),
                    "end": round(vad_seg["end"], 3),
                    "avg_logprob": avg_logprob,
                }
            )

        pbar.close()
        return {"segments": segments, "language": effective_language or "en"}


def load_mlx_model(
    whisper_arch: str,
    *,
    torch_device: str = "cpu",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
) -> MLXWhisperPipeline:
    """Build an :class:`MLXWhisperPipeline`.

    ``torch_device`` is the device for the *VAD* torch model only (mps/cpu); MLX
    Whisper inference uses the Apple GPU automatically.
    """
    model_path = _resolve_mlx_model(whisper_arch)
    logger.info("Loading MLX Whisper model: %s (VAD on %s)", model_path, torch_device)

    initial_prompt = (asr_options or {}).get("initial_prompt")

    default_vad_options = {"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        logger.info("Use manually assigned vad_model. vad_method is ignored.")
        resolved_vad = vad_model
    elif vad_method == "silero":
        resolved_vad = Silero(device=torch_device, **default_vad_options)
    elif vad_method == "pyannote":
        import torch

        resolved_vad = Pyannote(torch.device(torch_device), token=None, **default_vad_options)
    else:
        raise ValueError(f"Invalid vad_method: {vad_method}")

    return MLXWhisperPipeline(
        model_path=model_path,
        vad=resolved_vad,
        vad_params=default_vad_options,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
    )
