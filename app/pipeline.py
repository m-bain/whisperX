"""Model manager and per-job pipeline runner (CPU-only).

:class:`ModelManager` loads and caches multiple Whisper checkpoints (client-
selectable via :class:`WhisperModel`) while sharing one diarizer + align cache.
The heavy whisperx imports happen lazily inside the manager so importing this
module stays cheap.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# CPU-only for now. compute_type must be float32 on CPU (int8/float16 are CUDA).
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
DIARIZE_MODEL = os.environ.get(
    "WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
)
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "8"))


class WhisperModel(str, Enum):
    """Client-selectable Whisper checkpoints (the only names we will load).

    Subclasses ``str`` so members serialize to plain strings in JSON / SQLite
    and compare equal to their value. Construction validates: ``WhisperModel(x)``
    raises ``ValueError`` for anything not listed here, which is how we reject
    arbitrary HF-download strings coming from the client.
    """

    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    DISTIL_LARGE_V3 = "distil-large-v3"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]

    @classmethod
    def coerce(cls, name: str, default: "WhisperModel") -> "WhisperModel":
        """Return the matching member, or ``default`` if ``name`` is unknown."""
        try:
            return cls(name)
        except ValueError:
            return default


def _default_model() -> WhisperModel:
    return WhisperModel.coerce(os.environ.get("WHISPERX_MODEL", "small"), WhisperModel.SMALL)


# Seeds the initial active model; clients can switch at runtime (persisted by the store).
DEFAULT_MODEL = _default_model().value

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


class ModelManager:
    """Loads and caches multiple Whisper checkpoints; diarization + alignment shared.

    Only the Whisper ASR model is per-selection. The pyannote diarizer and the
    wav2vec2 align models (keyed by *language*, not Whisper model) are loaded
    once and shared across every cached Whisper model. Selecting a second model
    keeps the first in ``_asr`` (no eviction), so switching back is instant.
    """

    def __init__(self, active: Optional[str] = None):
        self._asr: dict[str, object] = {}          # model name -> FasterWhisperPipeline
        self._align_cache: dict = {}               # shared: lang -> (model, metadata)
        self._diarize = None                       # shared DiarizationPipeline or None
        self._diarize_loaded = False
        self._diarize_error: Optional[str] = None
        self._loading: set[str] = set()
        self._errors: dict[str, str] = {}
        self._active = WhisperModel.coerce(active or DEFAULT_MODEL, _default_model()).value
        self._lock = threading.Lock()              # guards dicts / _active / status reads
        self._load_lock = threading.Lock()         # held only during a heavy model load

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._asr

    # --- diarization (shared, loaded once) ------------------------------
    def ensure_diarize(self):
        """Load the pyannote diarizer once iff HF_TOKEN is set. Returns it or None."""
        if self._diarize_loaded:
            return self._diarize
        with self._load_lock:
            if self._diarize_loaded:
                return self._diarize
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning(
                    "HF_TOKEN not set: diarization disabled (transcribe + align only). "
                    "Set HF_TOKEN and accept the %s user agreement to enable speakers.",
                    DIARIZE_MODEL,
                )
                self._diarize_loaded = True
                return None
            try:
                from whisperx.diarize import DiarizationPipeline

                logger.info("Loading diarization model=%s", DIARIZE_MODEL)
                self._diarize = DiarizationPipeline(
                    model_name=DIARIZE_MODEL, token=hf_token, device=DEVICE
                )
            except Exception as exc:  # noqa: BLE001 - surface to status, don't crash boot
                self._diarize_error = str(exc)
                logger.exception("Diarization model load failed")
            finally:
                self._diarize_loaded = True
            return self._diarize

    # --- ASR (per-model, cached) ----------------------------------------
    def load_asr(self, name: str):
        """Load (or return cached) the Whisper pipeline for ``name``. Blocks while loading."""
        model = WhisperModel(name)  # raises ValueError on anything not whitelisted
        key = model.value
        with self._lock:
            if key in self._asr:
                return self._asr[key]
        with self._load_lock:
            with self._lock:
                if key in self._asr:
                    return self._asr[key]
                self._loading.add(key)
            try:
                import whisperx

                logger.info("Loading whisper model=%s on %s (%s)", key, DEVICE, COMPUTE_TYPE)
                pipe = whisperx.load_model(
                    key, device=DEVICE, compute_type=COMPUTE_TYPE, vad_method="pyannote"
                )
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._loading.discard(key)
                    self._errors[key] = str(exc)
                logger.exception("Whisper model=%s load failed", key)
                raise
            with self._lock:
                self._asr[key] = pipe
                self._loading.discard(key)
                self._errors.pop(key, None)
            return pipe

    def warm(self, name: str) -> None:
        """Load ``name`` in a background daemon thread (non-blocking)."""
        threading.Thread(
            target=self._warm, args=(name,), name=f"warm-{name}", daemon=True
        ).start()

    def _warm(self, name: str) -> None:
        try:
            self.load_asr(name)
        except Exception:  # noqa: BLE001 - already logged + recorded in _errors
            pass

    def set_active(self, name: str) -> dict:
        """Validate, set the active model, warm it in the background. Returns status()."""
        model = WhisperModel(name)
        with self._lock:
            self._active = model.value
        self.warm(model.value)
        return self.status()

    def bundle_for(self, name: str) -> ModelBundle:
        """Bundle the requested Whisper model with the shared diarizer + align cache.

        Loads the ASR model synchronously (called from the single job worker, so
        blocking is fine). The shared ``_align_cache`` dict means align models load
        once across all bundles; jobs are serialized so there is no race on it.
        """
        asr = self.load_asr(name)
        diarize = self.ensure_diarize()
        return ModelBundle(asr=asr, diarize=diarize, _align_cache=self._align_cache)

    def status(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "diarize": self._diarize is not None,
                "diarize_error": self._diarize_error,
                "models": [
                    {
                        "name": v,
                        "loaded": v in self._asr,
                        "loading": v in self._loading,
                        "error": self._errors.get(v),
                    }
                    for v in WhisperModel.values()
                ],
            }


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
