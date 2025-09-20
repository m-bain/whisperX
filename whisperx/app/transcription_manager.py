"""
Orchestrates the WhisperX transcription process for the Qt application.
Manages model lifecycle, worker coordination, and state management.
"""
import time

from PySide6.QtCore import QObject, Signal, QThreadPool
from typing import Dict, Any, Optional

from whisperx.app.app_config import AppConfig, TranscriptionConfig
from whisperx.app.transcription_workers import ModelLoaderWorker, TranscriptionWorker
from whisperx.app.whisperx_bridge import WhisperXBridge


class TranscriptionManager(QObject):
    """Manages the complete transcription workflow."""

    # Signals
    progress_updated = Signal(int)
    status_updated = Signal(str)
    transcription_completed = Signal(dict)
    error_occurred = Signal(str)
    models_loaded = Signal()

    def __init__(self, thread_pool: QThreadPool):
        super().__init__()
        self.thread_pool = thread_pool
        self.app_config = AppConfig()
        self.loaded_models: Optional[Dict[str, Any]] = None
        self.current_worker: Optional[Any] = None
        self._is_running = False

    def is_running(self) -> bool:
        """Check if transcription is currently running."""
        return self._is_running

    def start_transcription(self, audio_file: str, ui_config: Dict[str, Any]) -> None:
        """Start the transcription process."""
        if self._is_running:
            self.error_occurred.emit("Transcription already in progress")
            return

        try:
            # Update configuration from UI
            print("BEFORE: ", str(self.app_config))
            self.app_config.update_config(
                audio_file=audio_file,
                **ui_config
            )
            print("AFTER: ", str(self.app_config))

            config = self.app_config.get_current_config()
            print("AFTER ONLY CONGIF: ", str(config))

            self._is_running = True

            # If models need loading, start with model loader
            if self._models_need_loading(config):
                self._start_model_loading(config)
            else:
                # Use existing models for transcription
                self._start_transcription_worker(config)

        except Exception as e:
            self._is_running = False
            self.error_occurred.emit(f"Failed to start transcription: {str(e)}")

    def stop_transcription(self) -> None:
        """Stop current transcription process."""
        # Note: QRunnable doesn't have a built-in way to stop
        # This is a limitation we need to handle gracefully
        self._is_running = False
        self.status_updated.emit("Stopping transcription...")

    def _models_need_loading(self, config: TranscriptionConfig) -> bool:
        """Determine if models need to be loaded/reloaded."""
        if not self.loaded_models:
            return True

        # Check if configuration requires different models
        # This is simplified - in practice you'd compare model parameters
        return False

    def _start_model_loading(self, config: TranscriptionConfig) -> None:
        """Start model loading worker."""
        self.status_updated.emit("Preparing models...")

        worker = ModelLoaderWorker(config)
        worker.signals.progress_updated.connect(self.progress_updated)
        worker.signals.status_updated.connect(self.status_updated)
        worker.signals.models_loaded.connect(self._on_models_loaded)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(lambda: self._on_model_loading_finished(config))

        self.current_worker = worker
        self.thread_pool.start(worker)

    def _start_transcription_worker(self, config: TranscriptionConfig) -> None:
        """Start transcription worker."""
        worker = TranscriptionWorker(config, self.loaded_models)
        worker.signals.progress_updated.connect(self.progress_updated)
        worker.signals.status_updated.connect(self.status_updated)
        worker.signals.transcription_completed.connect(self._on_transcription_completed)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_transcription_finished)

        self.current_worker = worker
        self.thread_pool.start(worker)

    def start_transcription_direct(self, audio_file: str, ui_config: Dict[str, Any]) -> None:
        """Test: Run transcription directly on main thread."""
        if self._is_running:
            self.error_occurred.emit("Transcription already in progress")
            return

        try:
            # Update configuration
            self.app_config.update_config(audio_file=audio_file, **ui_config)
            config = self.app_config.get_current_config()
            self._is_running = True

            # Load models if needed (on main thread)
            if self._models_need_loading(config):
                bridge = WhisperXBridge()
                models = bridge.load_models(config, None, None)  # No callbacks
                self.loaded_models = models

            # Run transcription DIRECTLY on main thread (no worker thread)
            bridge = WhisperXBridge()
            start_time = time.time()
            result = bridge.transcribe_audio(
                config=config,
                models=self.loaded_models,
                progress_callback=None,  # No callbacks
                status_callback=None
            )
            end_time = time.time()

            print(f"Direct main thread transcription took: {end_time - start_time:.2f} seconds")

            # Emit results
            self.transcription_completed.emit(result)
            self._is_running = False

        except Exception as e:
            self._is_running = False
            self.error_occurred.emit(str(e))

    def _on_models_loaded(self, models: Dict[str, Any]) -> None:
        """Handle successful model loading."""
        self.loaded_models = models
        self.models_loaded.emit()

    def _on_model_loading_finished(self, config: TranscriptionConfig) -> None:
        """Handle model loading completion and start transcription."""
        if self._is_running:
            self._start_transcription_worker(config)

    def _on_transcription_completed(self, result: Dict[str, Any]) -> None:
        """Handle successful transcription completion."""
        self.transcription_completed.emit(result)

    def _on_transcription_finished(self) -> None:
        """Handle transcription worker completion."""
        self._is_running = False
        self.current_worker = None

    def _on_worker_error(self, error_message: str) -> None:
        """Handle worker errors."""
        self._is_running = False
        self.current_worker = None
        self.error_occurred.emit(error_message)

    def get_current_config(self) -> TranscriptionConfig:
        """Get current transcription configuration."""
        return self.app_config.get_current_config()

    def save_preset(self, name: str) -> None:
        """Save current configuration as preset."""
        self.app_config.save_preset(name)

    def load_preset(self, name: str) -> bool:
        """Load named preset."""
        return self.app_config.load_preset(name)

    def get_preset_names(self) -> list:
        """Get available preset names."""
        return self.app_config.get_preset_names()