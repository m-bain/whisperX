import torch


def get_default_device() -> str:
    """Return the name of the default accelerator available to PyTorch.

    The preference order is: cuda > npu > mps > cpu.  This allows WhisperX to
    automatically use non-CUDA accelerators such as Ascend NPU without falling
    back to CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.backends.mps.is_available():
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
