import sys
from typing import Optional, Sequence

import torch

from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class DeviceError(ValueError):
    """A requested device does not exist on this machine, or a stage was explicitly set to a device it cannot use."""


def mps_available() -> bool:
    return sys.platform == "darwin" and torch.backends.mps.is_available()


def auto_device() -> str:
    """The best device on this machine: mps > cpu on macOS, cuda > cpu on every other OS."""
    if sys.platform == "darwin":
        return "mps" if mps_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: str) -> str:
    """Resolve "auto" and check that a requested device exists on this machine (raises DeviceError if not).

    cuda (optionally indexed, e.g. cuda:1) and mps must exist here; other torch device types (e.g. xpu)
    pass through unchanged.
    """
    if device == "auto":
        return auto_device()
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError):
        raise DeviceError(f"Invalid device '{device}'") from None
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(f"Device '{device}' was requested but no CUDA GPU is available on this machine")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise DeviceError(f"Device '{device}' was requested but this machine has {torch.cuda.device_count()} CUDA GPU(s)")
    if parsed.type == "mps" and not mps_available():
        raise DeviceError(f"Device '{device}' was requested but no Apple GPU (MPS) is available on this machine")
    return device


def stage_device(
    stage: str, device: Optional[str], inherited: str, supported: Optional[Sequence[str]], fallback: str, hint: str
) -> str:
    """Pick the device for one pipeline stage.

    device is the stage's own setting (None = not set) and inherited the pipeline-wide one; supported lists
    the device types the stage can use (None = any torch device).
    A device set explicitly for the stage that the stage cannot use raises DeviceError; an inherited
    device (or "auto") moves the stage to fallback with a warning, so one --device works for the whole pipeline.
    """
    explicit = device is not None and device != "auto"
    resolved = resolve_device(device if device is not None else inherited)
    if supported is None or torch.device(resolved).type in supported:
        return resolved
    message = f"{stage} cannot run on {resolved}{hint}"
    if explicit:
        raise DeviceError(message)
    fallback = resolve_device(fallback)
    logger.warning(f"{message}; running it on {fallback}")
    return fallback


def ctranslate2_has_mps() -> bool:
    # Released CTranslate2 builds have no Metal backend; a build that adds one exposes get_mps_device_count
    import ctranslate2

    get_mps_device_count = getattr(ctranslate2, "get_mps_device_count", None)
    return get_mps_device_count is not None and get_mps_device_count() > 0
