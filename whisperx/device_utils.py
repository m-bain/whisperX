import sys
from typing import Callable, Optional

import torch

from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class DeviceError(ValueError):
    """A requested device does not exist on this machine, or a stage was explicitly set to a device it cannot use."""


def mps_available() -> bool:
    return sys.platform == "darwin" and torch.backends.mps.is_available()


def get_default_device() -> str:
    """Return the name of the best accelerator available to PyTorch.

    The preference order is: cuda > npu > mps > cpu.  This allows WhisperX to
    automatically use non-CUDA accelerators such as Ascend NPU or the Apple GPU
    without falling back to CPU.  A stage whose backend cannot use the picked
    device (e.g. CTranslate2 transcription on mps) runs on cpu with a warning.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if mps_available():
        return "mps"
    return "cpu"


def get_device_type(device):
    """Return the accelerator type of a torch.device or string device.

    Examples:
        >>> get_device_type("cuda:0")
        'cuda'
        >>> get_device_type("npu:0")
        'npu'
        >>> get_device_type("cpu")
        'cpu'
    """
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":")[0]
    return str(device)


def resolve_device(device: str) -> str:
    """Resolve "auto" and check that a requested device exists on this machine (raises DeviceError if not).

    "auto" resolves to get_default_device(). cuda (optionally indexed, e.g. cuda:1) and mps must exist here;
    other torch device types (e.g. npu, xpu) pass through unchanged.
    """
    if device == "auto":
        return get_default_device()
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
    stage: str,
    device: Optional[str],
    inherited: str,
    supports: Optional[Callable[[str], bool]],
    fallback: str,
    hint: str,
) -> str:
    """Pick the device for one pipeline stage.

    device is the stage's own setting (None = not set) and inherited the pipeline-wide one; supports tells
    whether the stage can run on a resolved device (None = any torch device).
    A device set explicitly for the stage that the stage cannot use raises DeviceError; an inherited
    device (or "auto") moves the stage to fallback with a warning, so one --device works for the whole pipeline.
    """
    explicit = device is not None and device != "auto"
    resolved = resolve_device(device if device is not None else inherited)
    if supports is None or supports(resolved):
        return resolved
    message = f"{stage} cannot run on {resolved}{hint}"
    if explicit:
        raise DeviceError(message)
    fallback = resolve_device(fallback)
    logger.warning(f"{message}; running it on {fallback}")
    return fallback
