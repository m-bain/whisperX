"""Model manager and per-job pipeline runner.

:class:`ModelManager` loads and caches multiple Whisper checkpoints (client-
selectable via :class:`WhisperModel`) while sharing one diarizer + align cache.
The heavy whisperx imports happen lazily inside the manager so importing this
module stays cheap.

The compute device (CPU or CUDA) is per-manager instance state, switchable at
runtime via :meth:`ModelManager.set_device` — which flushes every cached model
(ASR, align, diarizer) and reloads on the new device, since faster-whisper binds
the device at construction.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default device; overridable per-instance and switchable at runtime. compute_type
# is derived from the device (float32 on CPU, float16 on CUDA; ignored for mlx) —
# see _compute_for. "mlx" runs ASR on the Apple Silicon GPU via mlx-whisper and the
# torch stages (VAD/align/diarize) on mps — see _torch_device.
DEFAULT_DEVICE = os.environ.get("WHISPERX_DEVICE", "cpu")
DEVICES = ("cpu", "cuda", "mlx")
# Human-readable device names for the UI (status fragment, switcher).
DEVICE_LABELS = {"cpu": "CPU", "cuda": "GPU (CUDA)", "mlx": "Apple GPU (MLX)"}
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


def cuda_available() -> bool:
    """Whether a CUDA GPU is usable. torch import kept lazy (it is heavy)."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - any import/runtime failure means "no GPU"
        return False


def mlx_available() -> bool:
    """Whether the MLX (Apple Silicon GPU) ASR backend can run here."""
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_whisper") is not None
    )


def _compute_for(device: str) -> str:
    """Pick the compute type for a device: float16 on CUDA, float32 otherwise.

    Ignored on the mlx path (mlx-whisper picks its own precision)."""
    return "float16" if device == "cuda" else "float32"


def _torch_device(device: str) -> str:
    """Torch device for the non-ASR stages (VAD/align/diarize).

    For "mlx" the ASR runs on the Apple GPU but the torch stages run on mps
    (fallback cpu); cpu/cuda pass through unchanged.
    """
    if device != "mlx":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - older torch / non-Apple build
        pass
    return "cpu"

# Formats written to disk for download.
OUTPUT_FORMATS = ("srt", "vtt", "txt", "json")
# get_writer reads these keys directly (whisperx/utils.py).
WRITER_OPTIONS = {"max_line_width": None, "max_line_count": None, "highlight_words": False}


@dataclass
class ModelBundle:
    asr: object  # FasterWhisperPipeline
    device: str = DEFAULT_DEVICE  # device every model in this bundle lives on
    diarize: Optional[object] = None  # DiarizationPipeline or None when no HF token
    _align_cache: dict = field(default_factory=dict)  # lang -> (model, metadata)

    def align_model(self, language_code: str):
        """Lazily load and cache the wav2vec2 align model for a language."""
        import whisperx

        if language_code not in self._align_cache:
            logger.info("Loading align model for language=%s", language_code)
            self._align_cache[language_code] = whisperx.load_align_model(
                language_code=language_code, device=_torch_device(self.device)
            )
        return self._align_cache[language_code]


class ModelManager:
    """Loads and caches multiple Whisper checkpoints; diarization + alignment shared.

    Only the Whisper ASR model is per-selection. The pyannote diarizer and the
    wav2vec2 align models (keyed by *language*, not Whisper model) are loaded
    once and shared across every cached Whisper model. Selecting a second model
    keeps the first in ``_asr`` (no eviction), so switching back is instant.
    """

    def __init__(self, active: Optional[str] = None, device: Optional[str] = None):
        self._asr: dict[str, object] = {}          # model name -> FasterWhisperPipeline
        self._align_cache: dict = {}               # shared: lang -> (model, metadata)
        self._diarize = None                       # shared DiarizationPipeline or None
        self._diarize_loaded = False
        self._diarize_error: Optional[str] = None
        self._loading: set[str] = set()
        self._errors: dict[str, str] = {}
        self._active = WhisperModel.coerce(active or DEFAULT_MODEL, _default_model()).value
        self._device = self._resolve_device(device or DEFAULT_DEVICE)
        self._compute_type = _compute_for(self._device)
        self._lock = threading.Lock()              # guards dicts / _active / _device / status
        self._load_lock = threading.Lock()         # held only during a heavy model load

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Validate a device, falling back to cpu when cuda is unavailable."""
        if device not in DEVICES:
            logger.warning("Unknown device %r, falling back to cpu", device)
            return "cpu"
        if device == "cuda" and not cuda_available():
            logger.warning("device=cuda requested but no CUDA GPU available; using cpu")
            return "cpu"
        if device == "mlx" and not mlx_available():
            logger.warning("device=mlx requested but MLX is unavailable (needs Apple "
                           "Silicon + the 'mlx' extra); using cpu")
            return "cpu"
        return device

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    @property
    def device(self) -> str:
        with self._lock:
            return self._device

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

                diarize_device = _torch_device(self._device)
                logger.info("Loading diarization model=%s on %s", DIARIZE_MODEL, diarize_device)
                self._diarize = DiarizationPipeline(
                    model_name=DIARIZE_MODEL, token=hf_token, device=diarize_device
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

                logger.info("Loading whisper model=%s on %s (%s)", key, self._device, self._compute_type)
                pipe = whisperx.load_model(
                    key, device=self._device, compute_type=self._compute_type, vad_method="pyannote"
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
        with self._lock:
            device, align_cache = self._device, self._align_cache
        return ModelBundle(asr=asr, device=device, diarize=diarize, _align_cache=align_cache)

    def set_device(self, name: str) -> dict:
        """Switch the compute device: flush every cached model and reload on it.

        faster-whisper binds the device at construction, so changing it means
        discarding the ASR cache, the shared align cache, and the diarizer, then
        re-warming the active model on the new device. Callers must ensure no job
        is in flight (the server gates this on an idle queue); already-running
        jobs hold their own bundle refs, so the flush never pulls a model out
        from under them.
        """
        if name not in DEVICES:
            raise ValueError(f"Unknown device: {name}")
        if name == "cuda" and not cuda_available():
            raise ValueError("No CUDA GPU available")
        if name == "mlx" and not mlx_available():
            raise ValueError("MLX unavailable (needs Apple Silicon + the 'mlx' extra)")
        with self._lock:
            if name == self._device:
                return self._status_locked()
            leaving_cuda = self._device == "cuda"
            self._asr.clear()
            self._align_cache = {}
            self._diarize = None
            self._diarize_loaded = False
            self._diarize_error = None
            self._loading.clear()
            self._errors.clear()
            self._device = name
            self._compute_type = _compute_for(name)
        if leaving_cuda:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort GPU memory release
                pass
        logger.info("Device switched to %s (%s); reloading models", name, _compute_for(name))
        self.warm(self._active)
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        return {
            "active": self._active,
            "device": self._device,
            "cuda_available": cuda_available(),
            "mlx_available": mlx_available(),
            "diarize": self._diarize is not None,
            "diarize_error": self._diarize_error,
            # Token presence is independent of lazy-load timing: the diarizer
            # object is None until the background warm finishes, but that does
            # NOT mean diarization is unavailable. Drive the "no HF_TOKEN" toast
            # off this, not off `diarize`.
            "diarize_token": bool(
                os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            ),
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
    progress: Optional[Callable[[str], None]] = None,
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

    def _stage(name: str) -> None:
        if progress is not None:
            progress(name)

    _stage("decoding")
    logger.info("Decoding audio: %s", audio_path)
    audio = whisperx.load_audio(audio_path)  # 16kHz mono float32, decoded once
    duration = len(audio) / SAMPLE_RATE

    _stage("transcribing")
    logger.info("Transcribing (batch_size=%d)", BATCH_SIZE)
    result = bundle.asr.transcribe(audio, batch_size=BATCH_SIZE, language=language)

    lang = result["language"]
    _stage("loading_align")  # bundle.align_model may download/load a ~1.26 GB model
    align_model, align_meta = bundle.align_model(lang)
    _stage("aligning")
    logger.info("Aligning (language=%s)", lang)
    result = whisperx.align(
        result["segments"], align_model, align_meta, audio, _torch_device(bundle.device)
    )

    if bundle.diarize is not None:
        _stage("diarizing")
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
