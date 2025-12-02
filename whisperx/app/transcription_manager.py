"""
Orchestrates the WhisperX transcription process for the Qt application.
Manages model lifecycle, worker coordination, and state management.
"""
import time
import os
import gc
from PySide6.QtCore import QObject, Signal, QThreadPool
from typing import Dict, Any, Optional

from whisperx.app.app_config import AppConfig, TranscriptionConfig
from whisperx.app.transcription_workers import TranscriptionWorker


class TranscriptionManager(QObject):
    """
    Manages the complete transcription workflow.
    Simplified to delegate orchestration to WhisperX's built-in transcribe_with_callbacks().
    """

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
        self.cached_models: Optional[Dict[str, Any]] = None
        self.current_worker: Optional[Any] = None
        self._is_running = False

        # Tracking variables for metadata
        self._transcription_start_time: Optional[float] = None
        self._current_filepath: Optional[str] = None
        self._current_file_size: Optional[int] = None
        self._last_config: Optional[Dict[str, Any]] = None

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
            self.app_config.update_config(
                audio_file=audio_file,
                **ui_config
            )

            config = self.app_config.get_current_config()

            self._transcription_start_time = time.time()
            self._current_filepath = audio_file
            if os.path.exists(audio_file):
                self._current_file_size = os.path.getsize(audio_file)
            else:
                self._current_file_size = 0

            self._is_running = True

            # Check if models need to be cleared
            if self._should_clear_models(config):
                self._cleanup_models()

            # Start transcription worker
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

    def _should_clear_models(self, config: TranscriptionConfig) -> bool:
        """
        Determine if cached models should be cleared.
        Clear models only when critical parameters change.
        """
        if not self.cached_models:
            return False

        if not self._last_config:
            return False

        # Check if core parameters changed
        critical_params = ['model_name', 'device', 'device_index', 'compute_type']
        for param in critical_params:
            if getattr(config, param) != self._last_config.get(param):
                print(f"Model cache cleared: {param} changed from {self._last_config.get(param)} to {getattr(config, param)}")
                return True

        # Note: Language and alignment changes are handled by transcribe_with_callbacks()
        # It will reload only the necessary models automatically

        return False

    def _start_transcription_worker(self, config: TranscriptionConfig) -> None:
        """Start transcription worker."""
        # Store current config for future comparison
        self._last_config = {
            'model_name': config.model_name,
            'device': config.device,
            'device_index': config.device_index,
            'compute_type': config.compute_type,
            'language': config.language,
            'enable_alignment': config.enable_alignment,
            'enable_diarization': config.enable_diarization
        }

        worker = TranscriptionWorker(config, self.cached_models)
        worker.signals.progress_updated.connect(self.progress_updated)
        worker.signals.status_updated.connect(self.status_updated)
        worker.signals.models_loaded.connect(self._on_models_loaded)
        worker.signals.transcription_completed.connect(self._on_transcription_completed)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_transcription_finished)

        self.current_worker = worker
        self.thread_pool.start(worker)

    def _cleanup_models(self) -> None:
        """Clean up cached models to free memory."""
        if self.cached_models:
            print("Cleaning up cached models...")
            self.cached_models = None

            # Force garbage collection
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _on_models_loaded(self, models: Dict[str, Any]) -> None:
        """
        Handle models loaded/updated by worker.
        Merge new models with existing cache.
        """
        if self.cached_models is None:
            self.cached_models = models
        else:
            # Merge new models with existing cache
            self.cached_models.update(models)

        self.models_loaded.emit()

    def _on_transcription_completed(self, result: Dict[str, Any]) -> None:
        """Handle successful transcription completion."""
        if self._transcription_start_time:
            elapsed_time = time.time() - self._transcription_start_time
            result['elapsed_time'] = elapsed_time

        result['filepath'] = self._current_filepath
        result['file_size'] = self._current_file_size

        # Get current config as dict
        from dataclasses import asdict
        result['config'] = asdict(self.app_config.get_current_config())

        # Reset tracking variables
        self._transcription_start_time = None
        self._current_filepath = None
        self._current_file_size = None

        self.transcription_completed.emit(result)

    def _on_transcription_finished(self) -> None:
        """Handle transcription worker completion."""
        self._is_running = False
        self.current_worker = None

    def _on_worker_error(self, error_message: str) -> None:
        """Handle worker errors."""
        if self._transcription_start_time:
            elapsed_time = time.time() - self._transcription_start_time

            # Create error result for history tracking
            error_result = {
                'elapsed_time': elapsed_time,
                'filepath': self._current_filepath,
                'file_size': self._current_file_size,
                'error_message': error_message,
                'status': 'error',
                'formatted': {
                    'raw': '',
                    'timestamped': '',
                    'speakers': '',
                    'full': f'Error: {error_message}'
                }
            }

            # Get current config
            from dataclasses import asdict
            error_result['config'] = asdict(self.app_config.get_current_config())

        # Reset tracking variables
        self._transcription_start_time = None
        self._current_filepath = None
        self._current_file_size = None

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
