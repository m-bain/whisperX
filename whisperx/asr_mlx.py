"""Whisper ASR on the Apple GPU through MLX (mlx-whisper), used instead of faster-whisper on mps.

Uses the same VAD chunking as FasterWhisperPipeline and returns one segment per VAD chunk,
so alignment and diarization work unchanged. Install with `pip install whisperx[mlx]`.
"""
import os
from typing import List, Optional, Union

import numpy as np

from whisperx.asr import find_numeral_symbol_tokens, get_vad_segments, load_vad
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback, SingleSegment, TranscriptionResult

logger = get_logger(__name__)

# faster-whisper model names -> MLX-converted checkpoints on the Hugging Face Hub
MLX_MODELS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v1": "mlx-community/whisper-large-v1-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

# whisperx (faster-whisper) asr_options keys -> mlx_whisper.transcribe / DecodingOptions keywords
OPTION_NAMES = {
    "temperatures": "temperature",
    "compression_ratio_threshold": "compression_ratio_threshold",
    "log_prob_threshold": "logprob_threshold",
    "no_speech_threshold": "no_speech_threshold",
    "initial_prompt": "initial_prompt",
    "prefix": "prefix",
    "suppress_tokens": "suppress_tokens",
    "suppress_blank": "suppress_blank",
    "length_penalty": "length_penalty",
    "best_of": "best_of",
    "condition_on_previous_text": "condition_on_previous_text",
    "max_new_tokens": "sample_len",
}


def mlx_unsupported_options(asr_options: Optional[dict]) -> List[str]:
    """Options the caller set that change the transcript and that mlx-whisper cannot honor."""
    asr_options = asr_options or {}
    unsupported = []
    if (asr_options.get("beam_size") or 1) > 1:
        unsupported.append(f"--beam_size {asr_options['beam_size']} (beam search)")
    if asr_options.get("patience") is not None:
        unsupported.append(f"--patience {asr_options['patience']} (beam search)")
    if asr_options.get("hotwords"):
        unsupported.append("--hotwords (hotword prompting)")
    if asr_options.get("repetition_penalty") not in (None, 1):
        unsupported.append(f"repetition_penalty {asr_options['repetition_penalty']}")
    if asr_options.get("no_repeat_ngram_size"):
        unsupported.append(f"no_repeat_ngram_size {asr_options['no_repeat_ngram_size']}")
    return unsupported


def mlx_installed() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def mlx_repo_id(whisper_arch: str) -> str:
    """The MLX checkpoint for a model name: a known whisper name, or an explicit mlx-community/ repo."""
    if whisper_arch in MLX_MODELS:
        return MLX_MODELS[whisper_arch]
    if whisper_arch.startswith("mlx-community/"):
        return whisper_arch
    raise ValueError(
        f"No MLX checkpoint for model '{whisper_arch}' (needed to transcribe on mps). "
        f"Use one of {', '.join(MLX_MODELS)}, an mlx-community/ repo id or a local MLX model directory"
    )


def mlx_model_path(whisper_arch: str, download_root: Optional[str], local_files_only: bool) -> str:
    if os.path.isdir(whisper_arch):
        return whisper_arch
    from huggingface_hub import snapshot_download

    return snapshot_download(mlx_repo_id(whisper_arch), cache_dir=download_root, local_files_only=local_files_only)


def mlx_decode_options(asr_options: Optional[dict], compute_type: str) -> dict:
    options = {
        "fp16": compute_type != "float32",
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
    }
    asr_options = asr_options or {}
    for name, mlx_name in OPTION_NAMES.items():
        if asr_options.get(name) is not None:
            options[mlx_name] = asr_options[name]
    options["temperature"] = tuple(options["temperature"])
    return options


class MlxWhisperPipeline:
    """VAD + mlx-whisper transcription with the same interface and output as FasterWhisperPipeline."""

    def __init__(
        self,
        model_path: str,
        vad,
        vad_params: dict,
        options: dict,
        language: Optional[str],
        task: str,
        suppress_numerals: bool,
    ):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        self.model_path = model_path
        self.vad_model = vad
        self._vad_params = vad_params
        self.options = options
        self.preset_language = language
        self.task = task
        self.suppress_numerals = suppress_numerals
        # load (and cache) the model up front so download/load time is not counted as transcription
        self.dtype = mx.float16 if options["fp16"] else mx.float32
        self.model = ModelHolder.get_model(model_path, self.dtype)

    def detect_language(self, audio: np.ndarray) -> str:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

        if not self.model.is_multilingual:
            return "en"
        if audio.shape[0] < N_SAMPLES:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        mel = log_mel_spectrogram(audio[:N_SAMPLES], n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
        _, probs = self.model.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(self.dtype))
        language = max(probs, key=probs.get)
        logger.info(f"Detected language: {language} ({probs[language]:.2f}) in first 30s of audio")
        return language

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
        progress_callback: ProgressCallback = None,
        interleaved_context: bool = False,
    ) -> TranscriptionResult:
        import mlx_whisper

        if isinstance(audio, str):
            audio = load_audio(audio)
        if interleaved_context:
            logger.warning("interleaved_context is not supported when transcribing with MLX, ignoring it")
        if batch_size is not None and batch_size > 1:
            logger.warning(f"MLX transcription ignores --batch_size {batch_size} (it transcribes one VAD chunk at a time)")

        vad_segments = get_vad_segments(self.vad_model, self._vad_params, audio, chunk_size)
        language = language or self.preset_language or self.detect_language(audio)
        task = task or self.task

        options = dict(self.options)
        if self.suppress_numerals:
            from mlx_whisper.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(
                self.model.is_multilingual, num_languages=self.model.num_languages, language=language, task=task
            )
            logger.info("Suppressing numeral and symbol tokens")
            suppress_tokens = options.get("suppress_tokens", [-1])
            if isinstance(suppress_tokens, str):
                suppress_tokens = [int(t) for t in suppress_tokens.split(",")]
            options["suppress_tokens"] = list(set(find_numeral_symbol_tokens(tokenizer) + list(suppress_tokens)))

        segments: list[SingleSegment] = []
        total_segments = len(vad_segments)
        for idx, seg in enumerate(vad_segments):
            f1 = int(seg["start"] * SAMPLE_RATE)
            f2 = int(seg["end"] * SAMPLE_RATE)
            out = mlx_whisper.transcribe(
                audio[f1:f2],
                path_or_hf_repo=self.model_path,
                language=language,
                task=task,
                without_timestamps=True,
                **options,
            )
            text = out["text"]
            avg_logprob = float(np.mean([s["avg_logprob"] for s in out["segments"]])) if out["segments"] else 0.0

            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)
            if verbose:
                print(f"Transcript: [{round(seg['start'], 3)} --> {round(seg['end'], 3)}] {text}")
            segments.append(
                {
                    "text": text,
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "avg_logprob": avg_logprob,
                }
            )

        return {"segments": segments, "language": language}


def load_mlx_model(
    whisper_arch: str,
    vad_model,
    vad_method: Optional[str],
    vad_options: Optional[dict],
    vad_device: str,
    device_index: int,
    asr_options: Optional[dict],
    language: Optional[str],
    task: str,
    compute_type: str,
    threads: Optional[int],
    download_root: Optional[str],
    local_files_only: bool,
) -> MlxWhisperPipeline:
    # Check MLX and the model name before loading anything else, so a bad setup fails fast
    if not mlx_installed():
        raise ImportError("Transcribing on mps (the Apple GPU) uses MLX, which needs the mlx extra: pip install whisperx[mlx]")
    import mlx.core as mx

    if not os.path.isdir(whisper_arch):
        mlx_repo_id(whisper_arch)
    # Speed and hardware options that only apply to faster-whisper
    ignored = []
    if compute_type == "int8":
        ignored.append("--compute_type int8 (MLX runs in float16)")
    if threads:
        ignored.append(f"--threads {threads}")
    if device_index is not None:
        ignored.append(f"--device_index {device_index}")
    if ignored:
        logger.warning(f"MLX transcription ignores {', '.join(ignored)}")
    vad_model, vad_params = load_vad(vad_model, vad_method, vad_options, vad_device, device_index or 0)
    model_path = mlx_model_path(whisper_arch, download_root, local_files_only)
    logger.info(f"ASR engine: mlx, model: {model_path}, device: {mx.default_device()}")
    return MlxWhisperPipeline(
        model_path=model_path,
        vad=vad_model,
        vad_params=vad_params,
        options=mlx_decode_options(asr_options, compute_type),
        language=language,
        task=task,
        suppress_numerals=bool((asr_options or {}).get("suppress_numerals")),
    )
