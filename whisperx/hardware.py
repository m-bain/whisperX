"""
Hardware Detection and Capability Assessment for WhisperX
Provides comprehensive GPU detection, CUDA environment validation, and optimal settings recommendation.
"""

import platform
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    vram_total: int  # in MB
    vram_free: int  # in MB
    compute_capability: Tuple[int, int]
    cuda_cores: Optional[int] = None
    driver_version: Optional[str] = None


@dataclass
class CUDAEnvironment:
    """Information about CUDA installation and compatibility."""
    cuda_available: bool
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    pytorch_cuda_version: Optional[str] = None
    is_compatible: bool = False
    compatibility_issues: List[str] = None


@dataclass
class HardwareCapabilities:
    """Complete hardware assessment and recommendations."""
    gpus: List[GPUInfo]
    cuda_env: CUDAEnvironment
    cpu_cores: int
    system_ram: int  # in MB

    # Recommendations
    recommended_device: str
    recommended_device_index: int
    recommended_compute_type: str
    recommended_batch_size: int
    max_batch_size: int
    can_use_gpu: bool

    # Performance estimates
    estimated_speedup: float = 1.0
    warning_messages: List[str] = None


def detect_gpu_capabilities() -> List[GPUInfo]:
    """
    Detect all available GPUs and their capabilities.

    Returns:
        List of GPUInfo objects for each detected GPU.
    """
    gpus = []

    if not torch.cuda.is_available():
        logger.info("CUDA not available, no GPUs detected")
        return gpus

    try:
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} CUDA device(s)")

        for i in range(gpu_count):
            try:
                # Get basic GPU properties
                props = torch.cuda.get_device_properties(i)

                # Get memory info
                torch.cuda.set_device(i)
                vram_total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # Convert to MB
                vram_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) // (
                            1024 * 1024)

                # Extract compute capability
                compute_capability = (props.major, props.minor)

                # Estimate CUDA cores (approximate values for common architectures)
                cuda_cores = _estimate_cuda_cores(props.name, props.multi_processor_count, compute_capability)

                gpu_info = GPUInfo(
                    index=i,
                    name=props.name,
                    vram_total=vram_total,
                    vram_free=vram_free,
                    compute_capability=compute_capability,
                    cuda_cores=cuda_cores
                )

                gpus.append(gpu_info)
                logger.info(f"GPU {i}: {gpu_info.name}, VRAM: {vram_total}MB, Compute: {compute_capability}")

            except Exception as e:
                logger.warning(f"Failed to detect properties for GPU {i}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to detect GPUs: {e}")

    return gpus


def detect_cuda_environment() -> CUDAEnvironment:
    """
    Detect CUDA installation and validate environment compatibility.

    Returns:
        CUDAEnvironment object with detailed compatibility information.
    """
    cuda_env = CUDAEnvironment(
        cuda_available=torch.cuda.is_available(),
        compatibility_issues=[]
    )

    if not cuda_env.cuda_available:
        cuda_env.compatibility_issues.append("CUDA not available in PyTorch")
        return cuda_env

    try:
        # Get PyTorch CUDA version
        cuda_env.pytorch_cuda_version = torch.version.cuda
        logger.info(f"PyTorch CUDA version: {cuda_env.pytorch_cuda_version}")

        # Get system CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        import re
                        match = re.search(r'release (\d+\.\d+)', line)
                        if match:
                            cuda_env.cuda_version = match.group(1)
                            break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Could not detect system CUDA version (nvcc not found)")

        # Detect cuDNN version
        try:
            if hasattr(torch.backends.cudnn, 'version'):
                cudnn_version = torch.backends.cudnn.version()
                if cudnn_version:
                    cuda_env.cudnn_version = f"{cudnn_version // 1000}.{(cudnn_version % 1000) // 100}.{cudnn_version % 100}"
                    logger.info(f"cuDNN version: {cuda_env.cudnn_version}")
        except Exception as e:
            logger.warning(f"Could not detect cuDNN version: {e}")
            cuda_env.compatibility_issues.append("cuDNN version detection failed")

        # Check compatibility
        cuda_env.is_compatible = _validate_cuda_compatibility(cuda_env)

    except Exception as e:
        logger.error(f"Error detecting CUDA environment: {e}")
        cuda_env.compatibility_issues.append(f"CUDA environment detection failed: {e}")

    return cuda_env


def assess_optimal_settings(gpus: List[GPUInfo], cuda_env: CUDAEnvironment,
                            model_size: str = "small") -> HardwareCapabilities:
    """
    Assess hardware capabilities and provide optimal settings recommendations.

    Args:
        gpus: List of detected GPUs
        cuda_env: CUDA environment information
        model_size: Whisper model size to optimize for

    Returns:
        HardwareCapabilities with comprehensive recommendations
    """
    import psutil

    # Get system information
    cpu_cores = psutil.cpu_count(logical=False) or 4
    system_ram = psutil.virtual_memory().total // (1024 * 1024)  # Convert to MB

    warnings_list = []

    # Determine if GPU can be used
    can_use_gpu = (
            cuda_env.cuda_available and
            cuda_env.is_compatible and
            len(gpus) > 0 and
            any(gpu.vram_total > 1000 for gpu in gpus)  # At least 1GB VRAM
    )

    if can_use_gpu:
        # Select best GPU
        best_gpu = max(gpus, key=lambda g: (g.vram_total, g.cuda_cores or 0))
        recommended_device = "cuda"
        recommended_device_index = best_gpu.index

        # Determine compute type based on GPU capabilities
        if best_gpu.compute_capability >= (7, 0):  # Volta architecture and newer
            recommended_compute_type = "float16"
        elif best_gpu.compute_capability >= (6, 0):  # Pascal architecture
            if best_gpu.vram_total >= 8000:  # 8GB+
                recommended_compute_type = "float16"
            else:
                recommended_compute_type = "int8"
                warnings_list.append("Using int8 precision due to limited VRAM")
        else:  # Older architectures
            recommended_compute_type = "int8"
            warnings_list.append("Using int8 precision due to older GPU architecture")

        # Calculate batch size recommendations
        model_vram_requirements = _get_model_vram_requirements(model_size, recommended_compute_type)
        available_vram = best_gpu.vram_free - 500  # Reserve 500MB for system

        if available_vram < model_vram_requirements["min"]:
            # Fallback to CPU
            can_use_gpu = False
            warnings_list.append(
                f"Insufficient VRAM ({available_vram}MB available, {model_vram_requirements['min']}MB required)")
        else:
            # Calculate optimal batch size
            available_for_batches = available_vram - model_vram_requirements["base"]
            batch_vram_cost = model_vram_requirements["per_batch"]

            max_batch_size = min(32, max(1, available_for_batches // batch_vram_cost))
            recommended_batch_size = min(8, max_batch_size)  # Conservative default

            # Estimate performance speedup
            estimated_speedup = _estimate_gpu_speedup(best_gpu, recommended_compute_type)

    if not can_use_gpu:
        # CPU fallback
        recommended_device = "cpu"
        recommended_device_index = 0
        recommended_compute_type = "int8"  # Most efficient for CPU
        recommended_batch_size = min(4, max(1, cpu_cores // 2))  # Conservative CPU batching
        max_batch_size = min(8, cpu_cores)
        estimated_speedup = 1.0

        if len(gpus) > 0:
            warnings_list.append("GPU available but not suitable, falling back to CPU")

    return HardwareCapabilities(
        gpus=gpus,
        cuda_env=cuda_env,
        cpu_cores=cpu_cores,
        system_ram=system_ram,
        recommended_device=recommended_device,
        recommended_device_index=recommended_device_index,
        recommended_compute_type=recommended_compute_type,
        recommended_batch_size=recommended_batch_size,
        max_batch_size=max_batch_size,
        can_use_gpu=can_use_gpu,
        estimated_speedup=estimated_speedup,
        warning_messages=warnings_list
    )


def validate_hardware_for_model(model_size: str, device: str, compute_type: str,
                                batch_size: int) -> Tuple[bool, List[str]]:
    """
    Validate if the current hardware can handle the specified configuration.

    Args:
        model_size: Whisper model size
        device: Target device ("cpu" or "cuda")
        compute_type: Compute precision
        batch_size: Desired batch size

    Returns:
        Tuple of (is_valid, warning_messages)
    """
    warnings_list = []

    if device == "cuda":
        if not torch.cuda.is_available():
            return False, ["CUDA not available"]

        gpus = detect_gpu_capabilities()
        if not gpus:
            return False, ["No CUDA GPUs detected"]

        # Check VRAM requirements
        gpu = gpus[0]  # Use first GPU for validation
        requirements = _get_model_vram_requirements(model_size, compute_type)
        total_required = requirements["base"] + (requirements["per_batch"] * batch_size)

        if gpu.vram_total < total_required:
            return False, [f"Insufficient VRAM: {total_required}MB required, {gpu.vram_total}MB available"]

        if gpu.vram_free < total_required:
            warnings_list.append(f"VRAM may be insufficient: {total_required}MB required, {gpu.vram_free}MB free")

    return True, warnings_list


# Helper functions

def _estimate_cuda_cores(gpu_name: str, multiprocessor_count: int,
                         compute_capability: Tuple[int, int]) -> Optional[int]:
    """Estimate CUDA cores based on GPU architecture."""
    major, minor = compute_capability

    # CUDA cores per SM for different architectures
    cores_per_sm = {
        (3, 0): 192,  # Kepler GK10x
        (3, 2): 192,  # Kepler GK10x
        (3, 5): 192,  # Kepler GK11x
        (5, 0): 128,  # Maxwell GM10x
        (5, 2): 128,  # Maxwell GM20x
        (6, 0): 64,  # Pascal GP100
        (6, 1): 128,  # Pascal GP10x
        (7, 0): 64,  # Volta GV100
        (7, 5): 64,  # Turing TU10x
        (8, 0): 64,  # Ampere GA100
        (8, 6): 128,  # Ampere GA10x
        (8, 9): 128,  # Ada Lovelace AD10x
        (9, 0): 128,  # Hopper GH100
    }

    return cores_per_sm.get((major, minor), 128) * multiprocessor_count


def _validate_cuda_compatibility(cuda_env: CUDAEnvironment) -> bool:
    """Validate CUDA environment compatibility."""
    if not cuda_env.cuda_available:
        return False

    # Check for basic requirements
    try:
        # Test basic CUDA operations
        torch.cuda.is_available()
        torch.cuda.device_count()
        return True
    except Exception as e:
        cuda_env.compatibility_issues.append(f"CUDA compatibility test failed: {e}")
        return False


def _get_model_vram_requirements(model_size: str, compute_type: str) -> Dict[str, int]:
    """Get estimated VRAM requirements for different model configurations."""

    # Base VRAM requirements in MB (model weights + basic operations)
    base_requirements = {
        "tiny": {"float16": 200, "float32": 400, "int8": 100},
        "base": {"float16": 500, "float32": 1000, "int8": 250},
        "small": {"float16": 1000, "float32": 2000, "int8": 500},
        "medium": {"float16": 2500, "float32": 5000, "int8": 1250},
        "large": {"float16": 5000, "float32": 10000, "int8": 2500},
        "large-v2": {"float16": 5000, "float32": 10000, "int8": 2500},
        "large-v3": {"float16": 5000, "float32": 10000, "int8": 2500},
    }

    # Additional VRAM per batch item
    per_batch_requirements = {
        "tiny": {"float16": 50, "float32": 100, "int8": 25},
        "base": {"float16": 100, "float32": 200, "int8": 50},
        "small": {"float16": 150, "float32": 300, "int8": 75},
        "medium": {"float16": 250, "float32": 500, "int8": 125},
        "large": {"float16": 400, "float32": 800, "int8": 200},
        "large-v2": {"float16": 400, "float32": 800, "int8": 200},
        "large-v3": {"float16": 400, "float32": 800, "int8": 200},
    }

    model_key = model_size.lower()
    if model_key not in base_requirements:
        model_key = "small"  # Default fallback

    base = base_requirements[model_key].get(compute_type, base_requirements[model_key]["float16"])
    per_batch = per_batch_requirements[model_key].get(compute_type, per_batch_requirements[model_key]["float16"])

    return {
        "base": base,
        "per_batch": per_batch,
        "min": base + per_batch  # Minimum for batch_size=1
    }


def _estimate_gpu_speedup(gpu: GPUInfo, compute_type: str) -> float:
    """Estimate performance speedup compared to CPU."""
    base_speedup = 10.0  # Base GPU speedup

    # Adjust based on compute capability
    major, minor = gpu.compute_capability
    if major >= 8:  # Ampere and newer
        arch_multiplier = 1.5
    elif major >= 7:  # Volta/Turing
        arch_multiplier = 1.3
    elif major >= 6:  # Pascal
        arch_multiplier = 1.0
    else:  # Older
        arch_multiplier = 0.7

    # Adjust based on compute type
    if compute_type == "float16":
        precision_multiplier = 1.0
    elif compute_type == "int8":
        precision_multiplier = 1.2  # int8 can be faster
    else:  # float32
        precision_multiplier = 0.8

    # Adjust based on VRAM (more VRAM allows larger batches)
    if gpu.vram_total >= 8000:
        vram_multiplier = 1.2
    elif gpu.vram_total >= 4000:
        vram_multiplier = 1.0
    else:
        vram_multiplier = 0.8

    return base_speedup * arch_multiplier * precision_multiplier * vram_multiplier