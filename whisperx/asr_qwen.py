from typing import Any, List, Optional, Union

import numpy as np
import torch

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.log_utils import get_logger
from whisperx.schema import ProgressCallback, SingleSegment, TranscriptionResult
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
from whisperx.vads import Pyannote, Silero, Vad

logger = get_logger(__name__)

DEFAULT_QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"

WHISPER_TO_QWEN_LANGUAGE = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

QWEN_TO_WHISPER_LANGUAGE = {
    "arabic": "ar",
    "cantonese": "yue",
    "chinese": "zh",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "filipino": "tl",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "macedonian": "mk",
    "malay": "ms",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "vietnamese": "vi",
}

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def normalize_whisper_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    value = str(language).strip().lower()
    if not value:
        return None
    if value in LANGUAGES:
        return value
    if value in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[value]
    raise ValueError(f"Unsupported language value: {language}")


def whisper_language_code_to_qwen(language_code: Optional[str]) -> Optional[str]:
    code = normalize_whisper_language_code(language_code)
    if code is None:
        return None
    qwen_lang = WHISPER_TO_QWEN_LANGUAGE.get(code)
    if qwen_lang is not None:
        return qwen_lang
    # Fallback for future language additions.
    return LANGUAGES.get(code, code).title()


def qwen_language_to_whisper_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    first = str(language).split(",")[0].strip().lower()
    if not first:
        return None
    code = QWEN_TO_WHISPER_LANGUAGE.get(first)
    if code is not None:
        return code
    if first in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[first]
    return None


def _to_explicit_device(device: str, qwen_device_map: Optional[str]) -> str:
    if qwen_device_map:
        value = qwen_device_map.strip().lower()
        if value in {"cpu", "mps"} or value.startswith("cuda"):
            return qwen_device_map
    return device


def _uses_hf_device_map(qwen_device_map: Optional[str]) -> bool:
    if not qwen_device_map:
        return False
    lowered = qwen_device_map.strip().lower()
    if lowered in {"cpu", "mps"} or lowered.startswith("cuda"):
        return False
    return True


def _build_qwen_model_kwargs(
    qwen_dtype: str,
    qwen_device_map: Optional[str],
    local_files_only: bool,
    use_auth_token: Optional[Union[str, bool]],
) -> dict:
    kwargs: dict[str, Any] = {}
    if qwen_dtype in DTYPE_MAP:
        kwargs["torch_dtype"] = DTYPE_MAP[qwen_dtype]
    if qwen_device_map:
        trimmed = qwen_device_map.strip()
        lowered = trimmed.lower()
        if lowered not in {"cpu", "mps"} and not lowered.startswith("cuda"):
            kwargs["device_map"] = trimmed
    if local_files_only:
        kwargs["local_files_only"] = True
    if isinstance(use_auth_token, str) and use_auth_token:
        kwargs["token"] = use_auth_token
    return kwargs


class QwenAsrPipeline:
    def __init__(
        self,
        model,
        vad,
        vad_params: dict,
        language: Optional[str] = None,
    ):
        self.model = model
        self.vad_model = vad
        self._vad_params = vad_params
        self.preset_language = normalize_whisper_language_code(language)

    def _get_vad_preprocess(self):
        if hasattr(self.vad_model, "preprocess_audio") and hasattr(self.vad_model, "merge_chunks"):
            return self.vad_model.preprocess_audio, self.vad_model.merge_chunks
        return Pyannote.preprocess_audio, Pyannote.merge_chunks

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
    ) -> TranscriptionResult:
        del num_workers
        if task is not None and task != "transcribe":
            raise ValueError("Qwen backend currently supports only task='transcribe'")

        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_np = np.asarray(audio, dtype=np.float32)

        preprocess_audio, merge_chunks = self._get_vad_preprocess()
        waveform = preprocess_audio(audio_np)
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        if len(vad_segments) == 0:
            out_language = normalize_whisper_language_code(language) or self.preset_language or "en"
            return {"segments": [], "language": out_language}

        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if len(vad_segments) == 0:
            out_language = normalize_whisper_language_code(language) or self.preset_language or "en"
            return {"segments": [], "language": out_language}

        whisper_lang_code = normalize_whisper_language_code(language) or self.preset_language
        qwen_lang = whisper_language_code_to_qwen(whisper_lang_code)

        batch_size = len(vad_segments) if batch_size in {None, 0} else batch_size
        all_results: List[Any] = []
        for start_idx in range(0, len(vad_segments), batch_size):
            sub_segments = vad_segments[start_idx : start_idx + batch_size]
            sub_audio = []
            for seg in sub_segments:
                f1 = int(seg["start"] * SAMPLE_RATE)
                f2 = int(seg["end"] * SAMPLE_RATE)
                sub_audio.append((audio_np[f1:f2], SAMPLE_RATE))

            sub_result = self.model.transcribe(
                audio=sub_audio,
                language=qwen_lang,
                return_time_stamps=False,
            )
            all_results.extend(sub_result)

        segments: List[SingleSegment] = []
        detected_language_codes: List[str] = []
        total_segments = len(vad_segments)
        for idx, (seg, out) in enumerate(zip(vad_segments, all_results)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            if progress_callback is not None:
                progress_callback(((idx + 1) / total_segments) * 100)

            text = str(getattr(out, "text", "")).strip()
            if verbose:
                print(
                    f"Transcript: [{round(seg['start'], 3)} --> {round(seg['end'], 3)}] {text}"
                )

            qwen_detected = getattr(out, "language", None)
            detected_code = qwen_language_to_whisper_code(qwen_detected)
            if detected_code:
                detected_language_codes.append(detected_code)

            segments.append(
                {
                    "text": text,
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                }
            )

        if whisper_lang_code is not None:
            out_language = whisper_lang_code
        else:
            out_language = next((code for code in detected_language_codes if code), "en")

        return {"segments": segments, "language": out_language}


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="default",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[Vad] = None,
    vad_method: Optional[str] = "pyannote",
    vad_options: Optional[dict] = None,
    model=None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    use_auth_token: Optional[Union[str, bool]] = None,
    qwen_dtype: str = "float16",
    qwen_device_map: Optional[str] = None,
) -> QwenAsrPipeline:
    del device_index, compute_type, asr_options, download_root, threads
    if task != "transcribe":
        raise ValueError("Qwen backend currently supports only task='transcribe'")

    if model is not None:
        qwen_model = model
    else:
        try:
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            raise ImportError(
                "Qwen backend requires qwen-asr. Install with: pip install 'whisperx[qwen]'"
            ) from exc

        model_name = whisper_arch if whisper_arch else DEFAULT_QWEN_ASR_MODEL
        qwen_kwargs = _build_qwen_model_kwargs(
            qwen_dtype=qwen_dtype,
            qwen_device_map=qwen_device_map,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
        )
        qwen_model = Qwen3ASRModel.from_pretrained(model_name, **qwen_kwargs)

        if not _uses_hf_device_map(qwen_device_map):
            explicit_device = _to_explicit_device(device=device, qwen_device_map=qwen_device_map)
            try:
                qwen_model.model.to(explicit_device)
            except Exception as exc:
                logger.warning(
                    "Could not move Qwen ASR model to device %s (%s). Keeping default device.",
                    explicit_device,
                    exc,
                )

    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363,
    }
    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is None:
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            explicit_device = _to_explicit_device(device=device, qwen_device_map=qwen_device_map)
            if explicit_device.lower() == "mps":
                device_vad = "cpu"
            else:
                device_vad = explicit_device
            vad_model = Pyannote(torch.device(device_vad), token=None, **default_vad_options)
        else:
            raise ValueError(f"Invalid vad_method: {vad_method}")

    return QwenAsrPipeline(
        model=qwen_model,
        vad=vad_model,
        vad_params=default_vad_options,
        language=language,
    )
