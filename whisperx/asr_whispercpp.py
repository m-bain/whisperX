"""whisper.cpp ASR backend (via pywhispercpp).

A drop-in alternative to the faster-whisper path in :mod:`whisperx.asr`: same
``load_model`` entry contract (reached via ``device="whispercpp"`` there) and the
same ``transcribe`` return shape (``{"segments", "language"}``), so the
downstream align/diarize stages are untouched.

Structurally a sibling of :mod:`whisperx.asr_mlx`: it reuses the identical
VAD-segment serial loop, swapping mlx-whisper for a ``pywhispercpp.model.Model``.
whisper.cpp uses Metal automatically on Apple Silicon (and CPU elsewhere).
``pywhispercpp`` is imported lazily so ``import whisperx`` stays cheap and
cross-platform — nothing here touches it until a ``device="whispercpp"`` model is
actually loaded/run.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback, SingleSegment, TranscriptionResult
from whisperx.vads import Pyannote, Silero, Vad

logger = get_logger(__name__)

# Short Whisper names -> whisper.cpp ggml model names (pywhispercpp.constants.
# AVAILABLE_MODELS). Most are identity; whisper.cpp auto-downloads the ggml
# weights from the ggerganov/whisper.cpp HF repo on first use.
WHISPERCPP_MODEL_MAP = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "large-v3-turbo",
    # No distil ggml is published; fall back to large-v3 (distil is English-only
    # anyway, so this only matters for English jobs that opted into it).
    "distil-large-v3": "large-v3",
}


def _resolve_whispercpp_model(whisper_arch: str) -> str:
    """Map a short Whisper name to a whisper.cpp ggml model name."""
    if whisper_arch in WHISPERCPP_MODEL_MAP:
        mapped = WHISPERCPP_MODEL_MAP[whisper_arch]
        if mapped != whisper_arch:
            logger.info("Mapping model '%s' -> whisper.cpp ggml '%s'", whisper_arch, mapped)
        return mapped
    logger.warning(
        "Unknown model '%s' for whisper.cpp. Pass one of: %s",
        whisper_arch, list(WHISPERCPP_MODEL_MAP.keys()),
    )
    return whisper_arch


def _avg_logprob(segments: list) -> float:
    """Mean log-probability across whisper.cpp segments (best-effort; unused
    downstream except as a populated field)."""
    probs = [float(getattr(s, "probability", 0.0) or 0.0) for s in segments]
    probs = [p for p in probs if p > 0.0]
    if not probs:
        return 0.0
    return float(np.mean(np.log(probs)))


class WhisperCppPipeline:
    """WhisperX transcription pipeline using whisper.cpp (pywhispercpp).

    Mirrors the public surface of :class:`whisperx.asr.FasterWhisperPipeline`
    that the rest of the pipeline relies on: a ``transcribe`` method returning a
    :class:`~whisperx.schema.TranscriptionResult`. Like the MLX backend there is
    no batching — VAD segments are transcribed serially; whisper.cpp uses Metal
    on Apple Silicon automatically.
    """

    def __init__(
        self,
        model,  # pywhispercpp.model.Model (loaded once by load_whispercpp_model)
        vad,
        vad_params: dict,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ):
        self.model = model
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
        from tqdm import tqdm

        if isinstance(audio, str):
            audio = load_audio(audio)

        effective_language = language or self.preset_language
        effective_task = task or self.task
        translate = effective_task == "translate"

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

        # whisper.cpp needs an explicit language; auto-detect once up front when
        # the caller didn't pass one (mirrors the MLX backend's first-chunk detect).
        if effective_language is None and vad_segments:
            first = vad_segments[0]
            chunk = audio[int(first["start"] * SAMPLE_RATE):int(first["end"] * SAMPLE_RATE)]
            try:
                (detected, _prob), _all = self.model.auto_detect_language(chunk)
                effective_language = detected
                logger.info("whisper.cpp detected language: %s", detected)
            except Exception:  # noqa: BLE001 - fall back to whisper.cpp's own default
                effective_language = None

        segments: list[SingleSegment] = []
        total_segments = len(vad_segments)

        pbar = tqdm(total=total_segments, desc="Transcribing", unit="seg")
        for idx, vad_seg in enumerate(vad_segments):
            f1 = int(vad_seg["start"] * SAMPLE_RATE)
            f2 = int(vad_seg["end"] * SAMPLE_RATE)
            audio_chunk = audio[f1:f2]

            cpp_segments = self.model.transcribe(
                audio_chunk,
                language=effective_language or "auto",
                translate=translate,
                initial_prompt=self.initial_prompt or "",
                print_progress=False,
                print_realtime=False,
            )

            chunk_text = " ".join(s.text.strip() for s in cpp_segments).strip()
            avg_logprob = _avg_logprob(cpp_segments)

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


def load_whispercpp_model(
    whisper_arch: str,
    *,
    torch_device: str = "cpu",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    threads: int = 4,
) -> WhisperCppPipeline:
    """Build a :class:`WhisperCppPipeline`.

    ``torch_device`` is the device for the *VAD* torch model only (mps/cpu);
    whisper.cpp inference uses Metal on Apple Silicon automatically.
    """
    from pywhispercpp.model import Model  # lazy: optional dependency

    model_name = _resolve_whispercpp_model(whisper_arch)
    logger.info("Loading whisper.cpp model: %s (VAD on %s)", model_name, torch_device)

    # redirect_whispercpp_logs_to=False silences whisper.cpp's stderr chatter.
    model = Model(
        model_name,
        models_dir=download_root,
        redirect_whispercpp_logs_to=False,
        n_threads=threads,
    )

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

    return WhisperCppPipeline(
        model=model,
        vad=resolved_vad,
        vad_params=default_vad_options,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
    )
