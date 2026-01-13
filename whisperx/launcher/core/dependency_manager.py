"""
Dependency management for WhisperX Launcher.
Handles installation, updates, and switching between CPU and GPU versions.
"""
import os
import subprocess
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from enum import Enum

from .hardware_detection import HardwareDetector

logger = logging.getLogger(__name__)


class InstallType(Enum):
    """Installation types."""
    CPU = "cpu"
    CUDA_11_8 = "cuda11.8"
    CUDA_12_1 = "cuda12.1"
    ROCM_5_7 = "rocm5.7"

# Before running any pip command, create environment
env = os.environ.copy()
env['PYTHONNOUSERSITE'] = '1'  # Disable user site-packages



class DependencyManager:
    """Manages WhisperX dependencies and PyTorch installations."""

    # def __init__(self, install_dir: Path):
    #     """
    #     Initialize DependencyManager.
    #
    #     Args:
    #         install_dir: Root installation directory
    #     """
    #     self.install_dir = Path(install_dir)
    #     self.config_dir = Path.home() / '.whisperx_app'
    #     self.config_dir.mkdir(exist_ok=True)
    #
    #     self.config_file = self.config_dir / 'dependency_config.json'
    #     self.installed_lock = self.config_dir / 'installed.lock'
    #
    #     # Python executable (bundled with app)
    #     self.python_exe = self._find_python_executable()
    #
    #     # Load configuration
    #     self.config = self._load_config()
    def __init__(self, install_dir: Path):
        """Initialize DependencyManager."""
        self.install_dir = Path(install_dir)
        self.config_dir = Path.home() / '.whisperx_app'
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / 'dependency_config.json'
        self.installed_lock = self.config_dir / 'installed.lock'

        # Python executable (bundled with app)
        self.python_exe = self._find_python_executable()

        # Load configuration
        self.config = self._load_config()

        # NEW: Ensure pip is installed when using bundled Python
        # if getattr(sys, 'frozen', False):
        logger.info("Ensuring pip is installed in bundled Python...")
        if not self._ensure_pip_installed():
            logger.warning("Failed to ensure pip installation")
            # We don't raise here, will fail later during actual installation

    # def _find_python_executable(self) -> Path:
    #     """
    #     Find the bundled Python executable.
    #
    #     Returns:
    #         Path to Python executable
    #     """
    #     # When bundled with PyInstaller, Python is included
    #     if getattr(sys, 'frozen', False):
    #         # Running as bundled executable
    #         if sys.platform == 'win32':
    #             python_exe = self.install_dir / 'python' / 'python.exe'
    #         else:
    #             python_exe = self.install_dir / 'python' / 'bin' / 'python3'
    #     else:
    #         # Running from source (development)
    #         python_exe = Path(sys.executable)
    #
    #     return python_exe

    def _find_python_executable(self) -> Path:
        """
        Find the bundled Python executable.

        Returns:
            Path to Python executable
        """
        # When bundled with PyInstaller, Python is included
        if getattr(sys, 'frozen', False):
            # Running as bundled executable
            logger.info("Running as frozen executable, looking for bundled Python...")

            # The install_dir should be: C:\Program Files\WhisperX\SmartVoice\
            # Python should be at: C:\Program Files\WhisperX\SmartVoice\python\python.exe
            if sys.platform == 'win32':
                bundled_python = self.install_dir / 'python' / 'python.exe'
            else:
                bundled_python = self.install_dir / 'python' / 'bin' / 'python3'

            logger.info(f"Looking for bundled Python at: {bundled_python}")

            if bundled_python.exists():
                logger.info(f"Found bundled Python: {bundled_python}")
                return bundled_python
            else:
                # This is an error - bundled Python should exist
                error_msg = (
                    f"Bundled Python not found at: {bundled_python}\n"
                    f"Please reinstall the application."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        else:
            # Running from source (development)
            logger.info("Running from source, using virtual environment Python")
            python_exe = Path(sys.executable)
            logger.info(f"Using Python: {python_exe}")
            return python_exe

    def _ensure_pip_installed(self) -> bool:
        """
        Ensure pip is installed in bundled Python.

        Returns:
            True if pip is available, False otherwise
        """
        try:
            # Check if pip is already available
            logger.info(">>>>>>>>> Checking if pip is installed...")
            result = subprocess.run(
                [str(self.python_exe), '-m', 'pip', '--version'],
                capture_output=True,
                text=True,
                timeout=10,
                env=env
            )

            if result.returncode == 0:
                logger.info(f"pip is installed: {result.stdout.strip()}")
                # return True
            else:
                # pip not installed, try to install it
                logger.info("pip not found, installing...")

                # Look for get-pip.py
                get_pip = self.python_exe.parent / 'get-pip.py'

                if not get_pip.exists():
                    logger.error(f"get-pip.py not found at: {get_pip}")
                    return False

                # Install pip
                logger.info(f"Running: {self.python_exe} {get_pip}")
                result = subprocess.run(
                    [str(self.python_exe), str(get_pip)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env
                )

            if result.returncode == 0:
                logger.info("pip is ready!")

                # NEW: Install build dependencies
                logger.info("Installing build dependencies (setuptools, wheel)...")
                result = subprocess.run(
                    [str(self.python_exe), '-m', 'pip', 'install',
                     '--no-user',  # Don't install to user directory
                     'setuptools', 'wheel'],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env
                )

                if result.returncode == 0:
                    logger.info("Build dependencies installed successfully!")
                else:
                    logger.warning(f"Failed to install build dependencies: {result.stderr}")
                    # Don't fail - will try anyway

                return True

        except Exception as e:
            logger.error(f"Error ensuring pip: {e}")
            return False

    def is_installed(self) -> bool:
        """
        Check if dependencies are installed.

        Returns:
            True if installed, False otherwise
        """
        return self.installed_lock.exists()

    def get_installed_type(self) -> Optional[InstallType]:
        """
        Get the currently installed dependency type.

        Returns:
            InstallType or None if not installed
        """
        if not self.is_installed():
            return None

        install_type_str = self.config.get('install_type', 'cpu')
        try:
            return InstallType(install_type_str)
        except ValueError:
            return InstallType.CPU

    def install_dependencies(
        self,
        install_type: InstallType,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """
        Install WhisperX dependencies.

        Args:
            install_type: Type of installation (CPU/CUDA)
            progress_callback: Callback function(progress_percent, status_message)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Installing dependencies: {install_type.value}")

        try:
            # Step 1: Install PyTorch (50% of progress)
            if progress_callback:
                progress_callback(10, f"Installing PyTorch ({install_type.value})...")

            success = self._install_pytorch(install_type, progress_callback)
            if not success:
                logger.error("PyTorch installation failed")
                return False

            # Step 2: Install WhisperX and dependencies (40% of progress)
            if progress_callback:
                progress_callback(50, "Installing WhisperX dependencies...")

            success = self._install_whisperx_deps(progress_callback)
            if not success:
                logger.error("WhisperX dependencies installation failed")
                return False

            # Step 3: Verify installation (10% of progress)
            if progress_callback:
                progress_callback(90, "Verifying installation...")

            success = self._verify_installation(install_type)
            if not success:
                logger.error("Installation verification failed")
                return False

            # Mark as installed
            self._mark_installed(install_type)

            if progress_callback:
                progress_callback(100, "Installation complete!")

            logger.info("Dependencies installed successfully")
            return True

        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            if progress_callback:
                progress_callback(0, f"Installation failed: {str(e)}")
            return False

    def switch_installation_type(
        self,
        new_type: InstallType,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """
        Switch between CPU and GPU installations.

        Args:
            new_type: New installation type
            progress_callback: Progress callback function

        Returns:
            True if successful, False otherwise
        """
        current_type = self.get_installed_type()
        logger.info(f"Switching from {current_type} to {new_type.value}")

        try:
            # Uninstall PyTorch first
            if progress_callback:
                progress_callback(10, "Removing old PyTorch installation...")

            self._uninstall_pytorch()

            # Install new PyTorch version
            if progress_callback:
                progress_callback(30, f"Installing PyTorch ({new_type.value})...")

            success = self._install_pytorch(new_type, progress_callback)
            if not success:
                return False

            # Verify
            if progress_callback:
                progress_callback(90, "Verifying installation...")

            success = self._verify_installation(new_type)
            if not success:
                return False

            # Update config
            self._mark_installed(new_type)

            if progress_callback:
                progress_callback(100, "Switch complete!")

            logger.info(f"Successfully switched to {new_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch installation type: {e}")
            if progress_callback:
                progress_callback(0, f"Switch failed: {str(e)}")
            return False

    def _install_pytorch(
        self,
        install_type: InstallType,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """Install PyTorch based on installation type."""
        # Get install command
        install_cmd = HardwareDetector.get_pytorch_install_command(install_type.value)

        # Split command into args
        cmd_parts = install_cmd.split()
        # cmd_parts[0] = str(self.python_exe)  # Use bundled Python
        # cmd_parts[1] = '-m'  # python -m pip install ...
        packages_and_args = cmd_parts[2:]  # Everything after 'pip install'
        cmd = [str(self.python_exe), '-m', 'pip', 'install'] + packages_and_args

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            # Run pip install with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Monitor output
            for line in process.stdout:
                logger.debug(line.strip())
                if progress_callback and 'Downloading' in line:
                    # Update progress when downloading
                    progress_callback(20, "Downloading PyTorch...")

            process.wait()

            if process.returncode != 0:
                logger.error(f"PyTorch installation failed with code {process.returncode}")
                return False

            return True

        except Exception as e:
            logger.error(f"PyTorch installation error: {e}")
            return False

    def _install_whisperx_deps(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> bool:
        """Install WhisperX and its dependencies."""
        try:
            # Install from current directory (development) or from PyPI
            whisperx_path = Path(__file__).parent.parent.parent.parent

            if (whisperx_path / 'pyproject.toml').exists():
                # Install from local source
                logger.info("Installing WhisperX from local source")
                cmd = [str(self.python_exe), '-m', 'pip', 'install', '-e', str(whisperx_path)]
            else:
                # Install from PyPI
                logger.info("Installing WhisperX from PyPI")
                cmd = [str(self.python_exe), '-m', 'pip', 'install', 'whisperx']

            if progress_callback:
                progress_callback(60, "Installing WhisperX...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout,
                env = env
            )

            if result.returncode != 0:
                logger.error(f"WhisperX installation failed: {result.stderr}")
                return False

            # Install GUI dependencies
            if progress_callback:
                progress_callback(80, "Installing GUI dependencies...")

            gui_deps = [
                'PySide6>=6.5.0',
                'QT-PyQt-PySide-Custom-Widgets>=0.8.0',
                'requests>=2.28.0',
                'packaging>=21.0'
            ]

            cmd = [str(self.python_exe), '-m', 'pip', 'install'] + gui_deps

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode != 0:
                logger.error(f"GUI dependencies installation failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("Installation timed out")
            return False
        except Exception as e:
            logger.error(f"WhisperX installation error: {e}")
            return False

    def _uninstall_pytorch(self) -> bool:
        """Uninstall PyTorch packages."""
        try:
            logger.info("Uninstalling PyTorch...")

            cmd = [
                str(self.python_exe), '-m', 'pip', 'uninstall', '-y',
                'torch', 'torchvision', 'torchaudio'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            # Don't fail if packages weren't installed
            return True

        except Exception as e:
            logger.warning(f"PyTorch uninstall error: {e}")
            return True  # Continue anyway

    def _verify_installation(self, install_type: InstallType) -> bool:
        """
        Verify that dependencies are installed correctly.

        Args:
            install_type: Expected installation type

        Returns:
            True if verification passed
        """
        logger.info("Verifying installation...")

        try:
            # Test PyTorch import
            test_script = """
import sys
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
sys.exit(0)
"""
            result = subprocess.run(
                [str(self.python_exe), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"PyTorch verification failed: {result.stderr}")
                return False

            logger.info(f"PyTorch verification output:\n{result.stdout}")

            # Test WhisperX import
            test_script2 = """
import whisperx
print(f"WhisperX imported successfully")
"""
            result2 = subprocess.run(
                [str(self.python_exe), '-c', test_script2],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result2.returncode != 0:
                logger.error(f"WhisperX verification failed: {result2.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("Verification timed out")
            return False
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    def _mark_installed(self, install_type: InstallType) -> None:
        """
        Mark dependencies as installed.

        Args:
            install_type: Type of installation
        """
        # Update config
        self.config['install_type'] = install_type.value
        self.config['installed_at'] = str(Path.ctime(Path(__file__)))
        self._save_config()

        # Create lock file
        self.installed_lock.write_text(install_type.value)
        logger.info(f"Marked as installed: {install_type.value}")

    def clear_installation(self) -> None:
        """Clear installation markers (for reinstall)."""
        if self.installed_lock.exists():
            self.installed_lock.unlink()
        logger.info("Installation markers cleared")

    def _load_config(self) -> Dict[str, Any]:
        """Load dependency configuration."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def _save_config(self) -> None:
        """Save dependency configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def get_installation_info(self) -> Dict[str, Any]:
        """
        Get information about current installation.

        Returns:
            dict: Installation information
        """
        if not self.is_installed():
            return {
                'installed': False,
                'install_type': None
            }

        install_type = self.get_installed_type()

        # Get PyTorch version
        pytorch_version = "Unknown"
        cuda_available = False

        try:
            test_script = """
import torch
print(torch.__version__)
print(torch.cuda.is_available())
"""
            result = subprocess.run(
                [str(self.python_exe), '-c', test_script],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    pytorch_version = lines[0]
                    cuda_available = lines[1].lower() == 'true'

        except Exception as e:
            logger.warning(f"Failed to get PyTorch info: {e}")

        return {
            'installed': True,
            'install_type': install_type.value if install_type else None,
            'pytorch_version': pytorch_version,
            'cuda_available': cuda_available,
            'installed_at': self.config.get('installed_at', 'Unknown'),
            'config_dir': str(self.config_dir),
            'python_exe': str(self.python_exe)
        }


if __name__ == '__main__':
    # Test dependency manager
    logging.basicConfig(level=logging.INFO)

    install_dir = Path(__file__).parent.parent.parent.parent
    dm = DependencyManager(install_dir)

    print("\n=== Dependency Manager Test ===")
    print(f"Python executable: {dm.python_exe}")
    print(f"Installed: {dm.is_installed()}")

    if dm.is_installed():
        info = dm.get_installation_info()
        print(f"\nInstallation Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
