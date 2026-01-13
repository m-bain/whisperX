"""
WhisperX Launcher Application

Main launcher window with:
- Version management
- Dependency installation (CPU/GPU)
- Update checking
- Configuration
- Launch main SmartVoice application
"""

import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QComboBox,
    QGroupBox, QMessageBox, QTabWidget, QCheckBox, QSpacerItem,
    QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QIcon

from whisperx.launcher.core import (
    HardwareDetector,
    VersionManager,
    DependencyManager,
    InstallType
)
from whisperx.__version__ import __version__

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_install_directory() -> Path:
    """
    Get the correct installation directory based on execution context.

    Returns:
        Path to installation root directory
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled executable
        # sys.executable is the path to SmartVoiceLauncher.exe
        # Current: C:\Program Files\WhisperX\SmartVoice\Launcher\SmartVoiceLauncher.exe
        # Want:    C:\Program Files\WhisperX\SmartVoice\

        launcher_exe = Path(sys.executable)  # SmartVoiceLauncher.exe
        launcher_dir = launcher_exe.parent  # Launcher\
        install_root = launcher_dir.parent  # SmartVoice\

        logger.info(f"Running as frozen executable")
        logger.info(f"Install root: {install_root}")

        return install_root
    else:
        # Running from source (development)
        # If running: C:\...\whisperX\whisperx\launcher\launcher_main.py
        # Want:       C:\...\whisperX\

        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent

        logger.info(f"Running from source")
        logger.info(f"Project root: {project_root}")

        return project_root

class InstallWorker(QThread):
    """Background worker for dependency installation."""

    progress_updated = Signal(int, str)  # (progress_percent, status_message)
    installation_completed = Signal(bool)  # (success)

    def __init__(self, dependency_manager: DependencyManager, install_type: InstallType):
        super().__init__()
        self.dependency_manager = dependency_manager
        self.install_type = install_type

    def run(self):
        """Run installation in background."""
        try:
            success = self.dependency_manager.install_dependencies(
                self.install_type,
                progress_callback=self._progress_callback
            )
            self.installation_completed.emit(success)
        except Exception as e:
            logger.error(f"Installation error: {e}")
            self.installation_completed.emit(False)

    def _progress_callback(self, progress: int, message: str):
        """Forward progress updates to main thread."""
        self.progress_updated.emit(progress, message)


class SwitchWorker(QThread):
    """Background worker for switching between CPU/GPU."""

    progress_updated = Signal(int, str)
    switch_completed = Signal(bool)

    def __init__(self, dependency_manager: DependencyManager, new_type: InstallType):
        super().__init__()
        self.dependency_manager = dependency_manager
        self.new_type = new_type

    def run(self):
        """Run switching in background."""
        try:
            success = self.dependency_manager.switch_installation_type(
                self.new_type,
                progress_callback=self._progress_callback
            )
            self.switch_completed.emit(success)
        except Exception as e:
            logger.error(f"Switch error: {e}")
            self.switch_completed.emit(False)

    def _progress_callback(self, progress: int, message: str):
        """Forward progress updates to main thread."""
        self.progress_updated.emit(progress, message)


class LauncherWindow(QMainWindow):
    """Main launcher window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"SmartVoice Launcher v{__version__}")
        self.setMinimumSize(700, 600)

        # Initialize core components
        # self.install_dir = Path(__file__).parent.parent.parent
        self.install_dir = get_install_directory()
        self.hardware_detector = HardwareDetector()
        self.version_manager = VersionManager()
        self.dependency_manager = DependencyManager(self.install_dir)

        # Workers
        self.install_worker: Optional[InstallWorker] = None
        self.switch_worker: Optional[SwitchWorker] = None

        # Setup UI
        self.setup_ui()

        # Initial checks
        QTimer.singleShot(500, self.perform_initial_checks)

    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Header
        header_label = QLabel("WhisperX SmartVoice Launcher")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        # Version info
        version_label = QLabel(f"Version {__version__}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(version_label)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.setup_tab = self.create_setup_tab()
        self.version_tab = self.create_version_tab()
        self.settings_tab = self.create_settings_tab()

        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.version_tab, "Updates")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Launch button (prominent at bottom)
        self.launch_btn = QPushButton("Launch SmartVoice")
        self.launch_btn.setMinimumHeight(50)
        launch_font = QFont()
        launch_font.setPointSize(12)
        launch_font.setBold(True)
        self.launch_btn.setFont(launch_font)
        self.launch_btn.clicked.connect(self.launch_smartvoice)
        self.launch_btn.setEnabled(False)
        main_layout.addWidget(self.launch_btn)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_setup_tab(self) -> QWidget:
        """Create the setup/installation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # System information group
        sys_group = QGroupBox("System Information")
        sys_layout = QVBoxLayout()

        self.system_info_label = QLabel("Detecting system...")
        sys_layout.addWidget(self.system_info_label)

        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)

        # Installation group
        install_group = QGroupBox("Installation")
        install_layout = QVBoxLayout()

        # Installation status
        self.install_status_label = QLabel("Checking installation...")
        install_layout.addWidget(self.install_status_label)

        # Installation type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Installation Type:"))

        self.install_type_combo = QComboBox()
        self.install_type_combo.addItem("CPU Only (Recommended)", InstallType.CPU)
        self.install_type_combo.addItem("CUDA 11.8 (GPU)", InstallType.CUDA_11_8)
        self.install_type_combo.addItem("CUDA 12.1 (GPU)", InstallType.CUDA_12_1)
        self.install_type_combo.currentIndexChanged.connect(self.on_install_type_changed)
        type_layout.addWidget(self.install_type_combo)

        install_layout.addLayout(type_layout)

        # Install/Switch buttons
        button_layout = QHBoxLayout()

        self.install_btn = QPushButton("Install Dependencies")
        self.install_btn.clicked.connect(self.start_installation)
        button_layout.addWidget(self.install_btn)

        self.switch_btn = QPushButton("Switch to GPU")
        self.switch_btn.clicked.connect(self.switch_installation)
        self.switch_btn.setEnabled(False)
        button_layout.addWidget(self.switch_btn)

        self.reinstall_btn = QPushButton("Reinstall")
        self.reinstall_btn.clicked.connect(self.reinstall_dependencies)
        self.reinstall_btn.setEnabled(False)
        button_layout.addWidget(self.reinstall_btn)

        install_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        install_layout.addWidget(self.progress_bar)

        # Progress label
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        install_layout.addWidget(self.progress_label)

        install_group.setLayout(install_layout)
        layout.addWidget(install_group)

        # Installation log
        log_group = QGroupBox("Installation Log")
        log_layout = QVBoxLayout()

        self.install_log = QTextEdit()
        self.install_log.setReadOnly(True)
        self.install_log.setMaximumHeight(150)
        log_layout.addWidget(self.install_log)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

        return tab

    def create_version_tab(self) -> QWidget:
        """Create the version/updates tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Current version group
        current_group = QGroupBox("Current Version")
        current_layout = QVBoxLayout()

        self.current_version_label = QLabel(f"Installed: v{__version__}")
        current_layout.addWidget(self.current_version_label)

        current_group.setLayout(current_layout)
        layout.addWidget(current_group)

        # Update check group
        update_group = QGroupBox("Check for Updates")
        update_layout = QVBoxLayout()

        self.update_status_label = QLabel("Click 'Check for Updates' to see if a new version is available")
        self.update_status_label.setWordWrap(True)
        update_layout.addWidget(self.update_status_label)

        check_btn = QPushButton("Check for Updates")
        check_btn.clicked.connect(self.check_for_updates)
        update_layout.addWidget(check_btn)

        self.update_btn = QPushButton("Download and Install Update")
        self.update_btn.clicked.connect(self.download_update)
        self.update_btn.setEnabled(False)
        update_layout.addWidget(self.update_btn)

        update_group.setLayout(update_layout)
        layout.addWidget(update_group)

        # Release notes group
        notes_group = QGroupBox("Release Notes")
        notes_layout = QVBoxLayout()

        self.release_notes_text = QTextEdit()
        self.release_notes_text.setReadOnly(True)
        self.release_notes_text.setPlainText("No release notes to display")
        notes_layout.addWidget(self.release_notes_text)

        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)

        return tab

    def create_settings_tab(self) -> QWidget:
        """Create the settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Auto-check for updates
        self.auto_update_check = QCheckBox("Automatically check for updates on startup")
        self.auto_update_check.setChecked(True)
        layout.addWidget(self.auto_update_check)

        # Data location info
        data_group = QGroupBox("User Data Location")
        data_layout = QVBoxLayout()

        config_dir = Path.home() / '.whisperx_app'
        data_label = QLabel(f"Configuration and history are stored in:\n{config_dir}")
        data_label.setWordWrap(True)
        data_layout.addWidget(data_label)

        data_info = QLabel("This data is preserved during updates and reinstalls.")
        data_info.setWordWrap(True)
        data_layout.addWidget(data_info)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # About section
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout()

        about_text = QLabel(
            "WhisperX SmartVoice\n\n"
            "Automatic Speech Recognition with word-level timestamps and speaker diarization.\n\n"
            f"Version: {__version__}\n"
            "Created with WhisperX by Max Bain"
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)

        about_group.setLayout(about_layout)
        layout.addWidget(about_group)

        layout.addStretch()

        return tab

    def perform_initial_checks(self):
        """Perform initial system checks."""
        # Detect system
        system_info = self.hardware_detector.get_system_info()

        sys_text = f"OS: {system_info['os']}\n"
        sys_text += f"GPU: {system_info['gpu_type'].upper()}"

        if system_info['cuda_available']:
            sys_text += f" (CUDA {system_info['cuda_version']})"

        sys_text += f"\nRecommended: {system_info['recommended_install'].upper()}"

        self.system_info_label.setText(sys_text)

        # Check installation status
        self.check_installation_status()

        # Auto-check for updates if enabled
        if self.auto_update_check.isChecked():
            self.check_for_updates(silent=True)

    def check_installation_status(self):
        """Check and update installation status."""
        if self.dependency_manager.is_installed():
            info = self.dependency_manager.get_installation_info()

            status_text = f"✓ Installed: {info['install_type'].upper()}\n"
            status_text += f"PyTorch: {info['pytorch_version']}\n"
            status_text += f"CUDA Available: {info['cuda_available']}"

            self.install_status_label.setText(status_text)

            # Update button states
            self.install_btn.setEnabled(False)
            self.switch_btn.setEnabled(True)
            self.reinstall_btn.setEnabled(True)
            self.launch_btn.setEnabled(True)

            # Update switch button text
            current_type = self.dependency_manager.get_installed_type()
            if current_type == InstallType.CPU:
                self.switch_btn.setText("Switch to GPU")
            else:
                self.switch_btn.setText("Switch to CPU")

            self.log_message("Dependencies are installed and ready")

        else:
            self.install_status_label.setText("⚠ Dependencies not installed")
            self.install_btn.setEnabled(True)
            self.switch_btn.setEnabled(False)
            self.reinstall_btn.setEnabled(False)
            self.launch_btn.setEnabled(False)

            self.log_message("Please install dependencies to continue")

    def on_install_type_changed(self, index):
        """Handle installation type selection change."""
        install_type = self.install_type_combo.currentData()
        self.log_message(f"Selected installation type: {install_type.value}")

    def start_installation(self):
        """Start dependency installation."""
        install_type = self.install_type_combo.currentData()

        reply = QMessageBox.question(
            self,
            "Confirm Installation",
            f"This will install PyTorch ({install_type.value}) and WhisperX dependencies.\n\n"
            "This may take several minutes and download 2-3 GB of data.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.log_message(f"Starting installation: {install_type.value}")

        # Disable buttons
        self.install_btn.setEnabled(False)
        self.switch_btn.setEnabled(False)
        self.reinstall_btn.setEnabled(False)

        # Show progress
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)

        # Start worker
        self.install_worker = InstallWorker(self.dependency_manager, install_type)
        self.install_worker.progress_updated.connect(self.on_install_progress)
        self.install_worker.installation_completed.connect(self.on_installation_completed)
        self.install_worker.start()

    def switch_installation(self):
        """Switch between CPU and GPU installation."""
        current_type = self.dependency_manager.get_installed_type()

        # Determine new type
        if current_type == InstallType.CPU:
            # Show dialog to choose GPU version
            system_info = self.hardware_detector.get_system_info()
            recommended = system_info.get('recommended_install', 'cuda11.8')

            if recommended == 'cuda12.1':
                new_type = InstallType.CUDA_12_1
            else:
                new_type = InstallType.CUDA_11_8

            msg = f"Switch from CPU to GPU ({new_type.value})?"
        else:
            new_type = InstallType.CPU
            msg = "Switch from GPU to CPU?"

        reply = QMessageBox.question(
            self,
            "Confirm Switch",
            f"{msg}\n\nThis will reinstall PyTorch.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.log_message(f"Switching to: {new_type.value}")

        # Disable buttons
        self.install_btn.setEnabled(False)
        self.switch_btn.setEnabled(False)
        self.reinstall_btn.setEnabled(False)

        # Show progress
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)

        # Start worker
        self.switch_worker = SwitchWorker(self.dependency_manager, new_type)
        self.switch_worker.progress_updated.connect(self.on_install_progress)
        self.switch_worker.switch_completed.connect(self.on_installation_completed)
        self.switch_worker.start()

    def reinstall_dependencies(self):
        """Reinstall all dependencies."""
        reply = QMessageBox.question(
            self,
            "Confirm Reinstall",
            "This will completely reinstall all dependencies.\n\n"
            "Your user data and history will be preserved.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Clear installation markers
        self.dependency_manager.clear_installation()

        # Refresh status
        self.check_installation_status()

        self.log_message("Installation cleared. Please install dependencies again.")

    def on_install_progress(self, progress: int, message: str):
        """Handle installation progress updates."""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        self.log_message(f"[{progress}%] {message}")

    def on_installation_completed(self, success: bool):
        """Handle installation completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if success:
            self.log_message("✓ Installation completed successfully!")
            QMessageBox.information(
                self,
                "Installation Complete",
                "Dependencies have been installed successfully!\n\n"
                "You can now launch SmartVoice."
            )
        else:
            self.log_message("✗ Installation failed")
            QMessageBox.critical(
                self,
                "Installation Failed",
                "Failed to install dependencies.\n\n"
                "Check the installation log for details."
            )

        # Refresh status
        self.check_installation_status()

    def check_for_updates(self, silent: bool = False):
        """Check for application updates."""
        if not silent:
            self.update_status_label.setText("Checking for updates...")

        self.log_message("Checking for updates...")

        update_info = self.version_manager.check_for_updates(force=not silent)

        if 'error' in update_info:
            if not silent:
                self.update_status_label.setText(f"Error: {update_info['error']}")
                self.log_message(f"Update check failed: {update_info['error']}")
            return

        if update_info['update_available']:
            status = f"✓ Update available: v{update_info['latest_version']}\n"
            status += f"Current: v{update_info['current_version']}\n"
            status += f"Size: {self.version_manager.format_file_size(update_info['asset_size'])}"

            if update_info['requires_reinstall']:
                status += "\n⚠ This update requires reinstallation"

            self.update_status_label.setText(status)
            self.update_btn.setEnabled(True)

            # Show release notes
            self.release_notes_text.setPlainText(update_info['release_notes'])

            self.log_message(f"Update available: v{update_info['latest_version']}")

            if not silent:
                QMessageBox.information(
                    self,
                    "Update Available",
                    f"A new version is available: v{update_info['latest_version']}\n\n"
                    "Click 'Download and Install Update' to update."
                )

        else:
            self.update_status_label.setText("✓ You are up to date!")
            self.update_btn.setEnabled(False)
            self.log_message("No updates available")

            if not silent:
                QMessageBox.information(
                    self,
                    "Up to Date",
                    "You are running the latest version!"
                )

    def download_update(self):
        """Download and install update."""
        QMessageBox.information(
            self,
            "Update",
            "Update download will be implemented in the next phase.\n\n"
            "For now, please download the latest installer from GitHub releases."
        )

        # TODO: Implement update download and installation
        # This will be implemented after installer builds are working

    def launch_smartvoice(self):
        """Launch the main SmartVoice application."""
        self.log_message("Launching SmartVoice...")

        try:
            # Path to main SmartVoice application
            if getattr(sys, 'frozen', False):
                # Running as bundled executable
                smartvoice_exe = self.install_dir / 'smartvoice.exe'
            else:
                # Running from source (development)
                smartvoice_main = self.install_dir / 'whisperx' / 'appSmartVoice' / 'main.py'
                python_exe = self.dependency_manager.python_exe

                if smartvoice_main.exists():
                    subprocess.Popen([str(python_exe), str(smartvoice_main)])
                    self.log_message("SmartVoice launched successfully")

                    # Minimize launcher (don't close, so user can access it again)
                    self.showMinimized()
                    return
                else:
                    raise FileNotFoundError("SmartVoice main.py not found")

            # Bundled version
            if smartvoice_exe.exists():
                subprocess.Popen([str(smartvoice_exe)])
                self.log_message("SmartVoice launched successfully")
                self.showMinimized()
            else:
                raise FileNotFoundError("SmartVoice executable not found")

        except Exception as e:
            logger.error(f"Failed to launch SmartVoice: {e}")
            QMessageBox.critical(
                self,
                "Launch Failed",
                f"Failed to launch SmartVoice:\n\n{str(e)}"
            )

    def log_message(self, message: str):
        """Add message to installation log."""
        self.install_log.append(message)
        logger.info(message)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("SmartVoice Launcher")
    app.setOrganizationName("WhisperX")
    app.setApplicationVersion(__version__)

    # Create and show launcher window
    window = LauncherWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
