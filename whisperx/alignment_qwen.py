from typing import Iterable, Optional, Union

import numpy as np
import torch

from whisperx.asr_qwen import (
    _build_qwen_model_kwargs,
    _to_explicit_device,
    _uses_hf_device_map,
    whisper_language_code_to_qwen,
)
from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import AlignedTranscriptionResult, ProgressCallback, SingleSegment

logger = get_logger(__name__)

DEFAULT_QWEN_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir=None,
    model_cache_only: bool = False,
    qwen_dtype: str = "float16",
    qwen_device_map: Optional[str] = None,
    use_auth_token: Optional[str] = None,
):
    del model_dir
    try:
        from qwen_asr import Qwen3ForcedAligner
    except Exception as exc:
        raise ImportError(
            "Qwen alignment requires qwen-asr. Install with: pip install 'whisperx[qwen]'"
        ) from exc

    aligner_name = model_name or DEFAULT_QWEN_FORCED_ALIGNER
    kwargs = _build_qwen_model_kwargs(
        qwen_dtype=qwen_dtype,
        qwen_device_map=qwen_device_map,
        local_files_only=model_cache_only,
        use_auth_token=use_auth_token,
    )

    align_model = Qwen3ForcedAligner.from_pretrained(aligner_name, **kwargs)

    if not _uses_hf_device_map(qwen_device_map):
        explicit_device = _to_explicit_device(
            device=device, qwen_device_map=qwen_device_map
        )
        try:
            align_model.model.to(explicit_device)
        except Exception as exc:
            logger.warning(
                "Could not move Qwen forced aligner to device %s (%s). Keeping default device.",
                explicit_device,
                exc,
            )

    metadata = {
        "language": language_code,
        "type": "qwen_forced_aligner",
        "model_name": aligner_name,
    }
    return align_model, metadata


def align(
    transcript: Iterable[SingleSegment],
    model,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
) -> AlignedTranscriptionResult:
    del device, interpolate_method
    if return_char_alignments:
        raise NotImplementedError(
            "Qwen forced aligner path does not support character-level alignments."
        )
    transcript = list(transcript)

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    max_duration = audio.shape[1] / SAMPLE_RATE
    total_segments = len(transcript)
    aligned_segments = []
    word_segments = []

    whisper_language = align_model_metadata.get("language")
    qwen_language = whisper_language_code_to_qwen(whisper_language) or "English"

    for idx, segment in enumerate(transcript):
        if print_progress:
            base_progress = (
                ((idx + 1) / total_segments) * 100 if total_segments else 100.0
            )
            percent_complete = (
                (50 + base_progress / 2) if combined_progress else base_progress
            )
            print(f"Progress: {percent_complete:.2f}%...")
        if progress_callback is not None and total_segments:
            progress_callback(((idx + 1) / total_segments) * 100)

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        aligned_segment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }
        if avg_logprob is not None:
            aligned_segment["avg_logprob"] = avg_logprob

        if not text.strip():
            aligned_segments.append(aligned_segment)
            continue

        if t1 >= max_duration:
            logger.warning(
                'Failed to align segment ("%s"): start time exceeds audio duration, keeping original timestamps.',
                text,
            )
            aligned_segments.append(aligned_segment)
            continue

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(min(t2, max_duration) * SAMPLE_RATE)
        waveform_segment = audio[:, f1:f2].squeeze(0).cpu().numpy().astype(np.float32)

        if waveform_segment.size == 0:
            aligned_segments.append(aligned_segment)
            continue

        try:
            result_batch = model.align(
                audio=[(waveform_segment, SAMPLE_RATE)],
                text=[text],
                language=[qwen_language],
            )
            result = result_batch[0] if result_batch else []
        except Exception as exc:
            logger.warning(
                'Failed to align segment ("%s") with Qwen aligner: %s', text, exc
            )
            aligned_segments.append(aligned_segment)
            continue

        for item in result:
            word = str(item.text)
            start = round(float(item.start_time) + t1, 3)
            end = round(float(item.end_time) + t1, 3)
            if end < start:
                end = start
            word_entry = {
                "word": word,
                "start": start,
                "end": end,
                "score": 1.0,
            }
            aligned_segment["words"].append(word_entry)
            word_segments.append(dict(word_entry))

        aligned_segments.append(aligned_segment)

    word_segments.sort(key=lambda w: (w.get("start", 0.0), w.get("end", 0.0)))
    return {"segments": aligned_segments, "word_segments": word_segments}
