"""Model bundle and per-job pipeline runner (CPU-only).

Models are loaded once via :func:`load_bundle` and reused across jobs. The
heavy whisperx imports happen lazily inside ``load_bundle`` so importing this
module stays cheap.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# CPU-only for now. compute_type must be float32 on CPU (int8/float16 are CUDA).
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
MODEL_NAME = os.environ.get("WHISPERX_MODEL", "small")
DIARIZE_MODEL = os.environ.get(
    "WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
)
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "8"))

# Formats written to disk for download.
OUTPUT_FORMATS = ("srt", "vtt", "txt", "json")
# get_writer reads these keys directly (whisperx/utils.py).
WRITER_OPTIONS = {"max_line_width": None, "max_line_count": None, "highlight_words": False}


@dataclass
class ModelBundle:
    asr: object  # FasterWhisperPipeline
    diarize: Optional[object] = None  # DiarizationPipeline or None when no HF token
    _align_cache: dict = field(default_factory=dict)  # lang -> (model, metadata)

    def align_model(self, language_code: str):
        """Lazily load and cache the wav2vec2 align model for a language."""
        import whisperx

        if language_code not in self._align_cache:
            logger.info("Loading align model for language=%s", language_code)
            self._align_cache[language_code] = whisperx.load_align_model(
                language_code=language_code, device=DEVICE
            )
        return self._align_cache[language_code]


def load_bundle() -> ModelBundle:
    """Load ASR + diarization models once. Diarization is disabled if no HF_TOKEN."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    logger.info("Loading whisper model=%s on %s (%s)", MODEL_NAME, DEVICE, COMPUTE_TYPE)
    asr = whisperx.load_model(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad_method="pyannote",
    )

    diarize = None
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        logger.info("Loading diarization model=%s", DIARIZE_MODEL)
        diarize = DiarizationPipeline(
            model_name=DIARIZE_MODEL, token=hf_token, device=DEVICE
        )
    else:
        logger.warning(
            "HF_TOKEN not set: diarization disabled (transcribe + align only). "
            "Set HF_TOKEN and accept the %s user agreement to enable speakers.",
            DIARIZE_MODEL,
        )

    return ModelBundle(asr=asr, diarize=diarize)


def run_job(
    bundle: ModelBundle,
    audio_path: str,
    output_dir: str,
    *,
    language: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    artifact_basename: str = "transcript",
) -> dict:
    """Run the full pipeline for one audio file.

    Returns the transcription result dict (segments + word_segments, with
    ``speaker`` keys when diarization ran), plus ``duration`` (seconds) and
    ``num_segments``. Download artifacts are written to ``output_dir`` named
    ``<artifact_basename>.<fmt>`` so their paths are deterministic.
    """
    import whisperx
    from whisperx.audio import SAMPLE_RATE
    from whisperx.utils import get_writer

    logger.info("Decoding audio: %s", audio_path)
    audio = whisperx.load_audio(audio_path)  # 16kHz mono float32, decoded once
    duration = len(audio) / SAMPLE_RATE

    logger.info("Transcribing (batch_size=%d)", BATCH_SIZE)
    result = bundle.asr.transcribe(audio, batch_size=BATCH_SIZE, language=language)

    lang = result["language"]
    align_model, align_meta = bundle.align_model(lang)
    logger.info("Aligning (language=%s)", lang)
    result = whisperx.align(
        result["segments"], align_model, align_meta, audio, DEVICE
    )

    if bundle.diarize is not None:
        logger.info("Diarizing (min=%s max=%s)", min_speakers, max_speakers)
        diarize_df = bundle.diarize(
            audio, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_df, result)
        result["diarized"] = True
    else:
        result["diarized"] = False

    result["language"] = lang
    result["duration"] = duration
    result["num_segments"] = len(result.get("segments", []))

    # Write download artifacts with a fixed basename. The writer derives the
    # output name from the basename of the path we pass it, so pass the desired
    # stem (not the real audio path) to control naming.
    artifacts = {}
    name_stem = os.path.join(output_dir, artifact_basename)
    for fmt in OUTPUT_FORMATS:
        writer = get_writer(fmt, output_dir)
        writer(result, name_stem, WRITER_OPTIONS)
        artifacts[fmt] = os.path.join(output_dir, f"{artifact_basename}.{fmt}")
    result["artifacts"] = artifacts

    logger.info("Job complete: %d segments", result["num_segments"])
    return result
