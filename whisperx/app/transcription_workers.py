from PySide6.QtCore import QRunnable, QObject, Signal

import traceback

from typing import Optional, Dict, Any, Callable

from app_config import TranscriptionConfig
from whisperx_bridge import WhisperXBridge

class WorkerSignals(QObject):
    """Signals for worker thread communication."""

    # progress
    progress_updated = Signal(int)
    status_updated = Signal(str)

    # state / completion signals
    finished = Signal()
    error = Signal(str) # str = error message

    # data
    transcription_completed = Signal(dict)
    models_loaded = Signal(dict)

class ModelLoaderWorker(QRunnable):
    """Worker for loading WhisperX models in background."""

    def __init__(self, config: TranscriptionConfig):
        super().__init__()
        self.config = config
        self.signals = WorkerSignals()
        self.bridge = WhisperXBridge()

        # progress
        self._progress_callback = self._on_progress_update
        self._status_callback = self._on_status_update

    def _on_progress_update(self, progress: int) -> None:
        """Handle progress updates from WhisperX."""
        self.signals.progress_updated.emit(progress)

    def _on_status_update(self, status: str) -> None:
        """Handle status updates from WhisperX."""
        self.signals.status_updated.emit(status)

    def run(self):
        # Model loading
        try:
            self.signals.status_updated.emit("Loading model")
            self.signals.progress_updated.emit(0)

            models = self.bridge.load_models(
                config = self.config,
                progress_callback = self._progress_callback,
                status_callback = self._status_callback
            )

            self.signals.progress_updated.emit(100)
            self.signals.status_updated.emit("Models loaded successfully")
            self.signals.models_loaded.emit(models)
            self.signals.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

class TranscriptionWorker(QRunnable):
    """Worker for WhisperX transcription processing."""

    def __init__(self, config: TranscriptionConfig, models: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.models = models
        self.signals = WorkerSignals()
        self.bridge = WhisperXBridge()

        # progress trackung
        self._progress_callback = self._on_progress_update
        self._status_callback = self._on_status_update

    def _on_progress_update(self, progress: int) -> None:
        self.signals.progress_updated.emit(progress)

    def _on_status_update(self, status: str) -> None:
        self.signals.status_updated.emit(status)

    def run(self):
        # Execute transcription process
        try:
            if not self.config.audio_file:
                raise ValueError("No audio file specified!")

            self.signals.status_updated.emit("Starting transcription...")
            self.signals.progress_updated.emit(0)

            # perform transcription
            result = self.bridge.transcribe_audio(
                config=self.config,
                models=self.models,
                progress_callback=self._progress_callback,
                status_callback=self._status_callback
            )

            self.signals.progress_updated.emit(100)
            self.signals.status_updated.emit("Transcription completed")
            self.signals.transcription_completed.emit(result)
            self.signals.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

