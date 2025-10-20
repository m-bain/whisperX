"""
Orchestrates the WhisperX transcription process for the Qt application.
Manages model lifecycle, worker coordination, and state management.
"""
import time
import os
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

        self._transcription_start_time: Optional[float] = None
        self._current_filepath: Optional[str] = None
        self._current_file_size: Optional[int] = None

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

            self._transcription_start_time = time.time()
            self._current_filepath = audio_file
            if os.path.exists(audio_file):
                self._current_file_size = os.path.getsize(audio_file)
            else:
                self._current_file_size = 0

            self._is_running = True

            # If models need loading, start with model loader
            if self._models_need_loading(config):
                self._cleanup_models()
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

        # Check if ASR model parameters changed
        if 'asr' in self.loaded_models:
            current_asr = self.loaded_models['asr']

            # Check core model parameters
            if (hasattr(current_asr, 'model') and
                    hasattr(current_asr.model, 'model_path')):
                # Extract current model name from path
                current_model = current_asr.model.model_path.split('/')[-1]
                if current_model != config.model_name:
                    print(f"Model changed: {current_model} -> {config.model_name}")
                    return True

            # Check if language changed (most important for your issue)
            if hasattr(current_asr, 'preset_language'):
                if current_asr.preset_language != config.language:
                    print(f"Language changed: {current_asr.preset_language} -> {config.language}")
                    return True

            # Check device changes
            if hasattr(current_asr, 'device'):
                current_device = str(current_asr.device)
                config_device = config.device
                if current_device != config_device:
                    print(f"Device changed: {current_device} -> {config_device}")
                    return True

        # Check if alignment model language changed
        if 'alignment' in self.loaded_models and config.enable_alignment:
            current_align_lang = self.loaded_models['alignment']['metadata'].get('language', 'en')
            target_lang = config.language or 'en'
            if current_align_lang != target_lang:
                print(f"Alignment language changed: {current_align_lang} -> {target_lang}")
                return True

        # If diarization setting changed
        if config.enable_diarization and 'diarization' not in self.loaded_models:
            return True
        elif not config.enable_diarization and 'diarization' in self.loaded_models:
            return True

        return False

    def _store_model_config(self, config: TranscriptionConfig) -> None:
        """Store the configuration used to load current models."""
        self._current_model_config = {
            'model_name': config.model_name,
            'language': config.language,
            'device': config.device,
            'device_index': config.device_index,
            'compute_type': config.compute_type,
            'enable_alignment': config.enable_alignment,
            'enable_diarization': config.enable_diarization
        }

    def _get_stored_model_config(self) -> dict:
        """Get the stored model configuration."""
        return getattr(self, '_current_model_config', {})

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

    def _cleanup_models(self) -> None:
        """Clean up loaded models to force reload."""
        if self.loaded_models:
            print("Cleaning up cached models...")
            # Cleanup GPU memory if using CUDA
            if 'asr' in self.loaded_models:
                del self.loaded_models['asr']
            if 'alignment' in self.loaded_models:
                del self.loaded_models['alignment']
            if 'diarization' in self.loaded_models:
                del self.loaded_models['diarization']

            self.loaded_models = None

            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _on_models_loaded(self, models: Dict[str, Any]) -> None:
        """Handle successful model loading."""
        self.loaded_models = models
        # Store the config used for these models
        current_config = self.app_config.get_current_config()
        self._store_model_config(current_config)
        self.models_loaded.emit()

    def _on_model_loading_finished(self, config: TranscriptionConfig) -> None:
        """Handle model loading completion and start transcription."""
        if self._is_running:
            self._start_transcription_worker(config)

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

            # Could emit a separate error completion signal if needed
            # For now, just emit error

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