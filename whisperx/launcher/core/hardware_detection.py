"""
Hardware detection module for WhisperX Launcher.
Detects GPU capabilities and determines appropriate PyTorch installation.
"""

import subprocess
import platform
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detect system hardware and determine optimal dependencies."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            dict: {
                'os': 'Windows' | 'Linux' | 'Darwin',
                'os_version': str,
                'architecture': 'x86_64' | 'arm64',
                'gpu_type': 'nvidia' | 'amd' | 'cpu',
                'cuda_available': bool,
                'cuda_version': str | None,
                'gpu_name': str | None,
                'recommended_install': 'cpu' | 'cuda11.8' | 'cuda12.1'
            }
        """
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
        }

        # Detect GPU
        gpu_info = HardwareDetector.detect_gpu()
        info.update(gpu_info)

        # Determine recommended installation
        info['recommended_install'] = HardwareDetector._determine_recommended_install(info)

        return info

    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """
        Detect GPU type and CUDA availability.

        Returns:
            dict: {
                'gpu_type': str,
                'cuda_available': bool,
                'cuda_version': str | None,
                'gpu_name': str | None
            }
        """
        # Try NVIDIA first
        nvidia_info = HardwareDetector._detect_nvidia()
        if nvidia_info['cuda_available']:
            return nvidia_info

        # Try AMD
        amd_info = HardwareDetector._detect_amd()
        if amd_info['gpu_type'] == 'amd':
            return amd_info

        # Fallback to CPU
        return {
            'gpu_type': 'cpu',
            'cuda_available': False,
            'cuda_version': None,
            'gpu_name': None
        }

    @staticmethod
    def _detect_nvidia() -> Dict[str, Any]:
        """Detect NVIDIA GPU and CUDA version."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                gpu_name = output.split(',')[0].strip() if ',' in output else 'NVIDIA GPU'

                # Get CUDA version from nvidia-smi
                cuda_result = subprocess.run(
                    ['nvidia-smi'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == 'Windows' else 0
                )

                cuda_version = None
                if cuda_result.returncode == 0:
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', cuda_result.stdout)
                    if match:
                        cuda_version = match.group(1)

                logger.info(f"Detected NVIDIA GPU: {gpu_name}, CUDA: {cuda_version}")

                return {
                    'gpu_type': 'nvidia',
                    'cuda_available': True,
                    'cuda_version': cuda_version,
                    'gpu_name': gpu_name
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"NVIDIA detection failed: {e}")

        return {
            'gpu_type': 'cpu',
            'cuda_available': False,
            'cuda_version': None,
            'gpu_name': None
        }

    @staticmethod
    def _detect_amd() -> Dict[str, Any]:
        """Detect AMD GPU (ROCm support)."""
        try:
            # Try rocm-smi for Linux
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpu_name = result.stdout.strip() or 'AMD GPU'
                logger.info(f"Detected AMD GPU: {gpu_name}")
                return {
                    'gpu_type': 'amd',
                    'cuda_available': False,
                    'cuda_version': None,
                    'gpu_name': gpu_name
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"AMD detection failed: {e}")

        return {
            'gpu_type': 'cpu',
            'cuda_available': False,
            'cuda_version': None,
            'gpu_name': None
        }

    @staticmethod
    def _determine_recommended_install(system_info: Dict[str, Any]) -> str:
        """
        Determine recommended PyTorch installation type.

        Args:
            system_info: System information dictionary

        Returns:
            'cpu' | 'cuda11.8' | 'cuda12.1' | 'rocm5.7'
        """
        if not system_info.get('cuda_available'):
            return 'cpu'

        cuda_version = system_info.get('cuda_version')
        if cuda_version:
            cuda_major = float(cuda_version.split('.')[0])

            # CUDA 12.x -> cuda12.1
            if cuda_major >= 12:
                return 'cuda12.1'
            # CUDA 11.x -> cuda11.8
            elif cuda_major >= 11:
                return 'cuda11.8'

        # Default to CUDA 11.8 if CUDA detected but version unknown
        return 'cuda11.8'

    @staticmethod
    def can_run_gpu() -> bool:
        """Quick check if GPU is available."""
        return HardwareDetector.detect_gpu()['cuda_available']

    @staticmethod
    def get_pytorch_install_command(install_type: str = 'cpu', python_version: str = '3.10') -> str:
        """
        Get the pip install command for PyTorch based on install type.

        Args:
            install_type: 'cpu', 'cuda11.8', or 'cuda12.1'
            python_version: Python version (e.g., '3.9')

        Returns:
            pip install command string
        """
        base_packages = "torch torchvision torchaudio"

        if install_type == 'cpu':
            return f"pip install {base_packages} --index-url https://download.pytorch.org/whl/cpu"
        elif install_type == 'cuda11.8':
            return f"pip install {base_packages} --index-url https://download.pytorch.org/whl/cu118"
        elif install_type == 'cuda12.1':
            return f"pip install {base_packages} --index-url https://download.pytorch.org/whl/cu121"
        elif install_type == 'rocm5.7':
            return f"pip install {base_packages} --index-url https://download.pytorch.org/whl/rocm5.7"
        else:
            # Fallback to CPU
            return f"pip install {base_packages} --index-url https://download.pytorch.org/whl/cpu"


if __name__ == '__main__':
    # Test the hardware detection
    logging.basicConfig(level=logging.INFO)

    detector = HardwareDetector()
    info = detector.get_system_info()

    print("\n=== System Information ===")
    print(f"OS: {info['os']} {info['os_version']}")
    print(f"Architecture: {info['architecture']}")
    print(f"GPU Type: {info['gpu_type']}")
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_version']:
        print(f"CUDA Version: {info['cuda_version']}")
    if info['gpu_name']:
        print(f"GPU Name: {info['gpu_name']}")
    print(f"Recommended Install: {info['recommended_install']}")
    print(f"\nPyTorch Install Command:\n{detector.get_pytorch_install_command(info['recommended_install'])}")
