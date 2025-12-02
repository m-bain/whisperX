"""
Qt workers for background transcription processing.
Thin wrappers around WhisperX's transcribe_with_callbacks() function.
"""
from PySide6.QtCore import QRunnable, QObject, Signal

import traceback
import time
from typing import Optional, Dict, Any

from whisperx.app.app_config import TranscriptionConfig


class WorkerSignals(QObject):
    """Signals for worker thread communication."""

    # Progress and status
    progress_updated = Signal(int)
    status_updated = Signal(str)

    # State / completion signals
    finished = Signal()
    error = Signal(str)  # str = error message

    # Data
    transcription_completed = Signal(dict)
    models_loaded = Signal(dict)


class TranscriptionWorker(QRunnable):
    """
    Worker for WhisperX transcription processing.
    Uses WhisperX's built-in transcribe_with_callbacks() function.
    """

    def __init__(self, config: TranscriptionConfig, cached_models: Optional[Dict] = None):
        super().__init__()
        self.config = config
        self.cached_models = cached_models
        self.signals = WorkerSignals()

        # Progress tracking
        self._last_progress_time = 0
        self._last_progress_value = -1

    def _on_progress_update(self, progress: int) -> None:
        """Throttle progress updates to avoid overwhelming the UI."""
        # Only emit progress updates every 2% or every 500ms
        current_time = time.time()
        if (progress - self._last_progress_value >= 2 or
            current_time - self._last_progress_time >= 0.5):
            self.signals.progress_updated.emit(progress)
            self._last_progress_time = current_time
            self._last_progress_value = progress

    def _on_status_update(self, status: str) -> None:
        """Forward status messages to UI."""
        self.signals.status_updated.emit(status)

    def run(self):
        """
        Execute transcription using WhisperX's built-in function.
        Uses lazy import to avoid loading heavy dependencies at app startup.
        """
        try:
            if not self.config.audio_file:
                raise ValueError("No audio file specified!")

            self.signals.status_updated.emit("Starting transcription...")
            self.signals.progress_updated.emit(0)

            # LAZY IMPORT - only loads when transcription actually starts
            # This is critical for fast startup time
            from whisperx.transcribe import transcribe_with_callbacks
            from whisperx.utils import format_timestamp

            # Perform transcription using WhisperX's built-in function
            result = transcribe_with_callbacks(
                audio_file=self.config.audio_file,
                model_name=self.config.model_name,
                device=self.config.device,
                device_index=self.config.device_index,
                compute_type=self.config.compute_type,
                language=self.config.language,
                enable_alignment=self.config.enable_alignment,
                enable_diarization=self.config.enable_diarization,
                batch_size=self.config.batch_size,
                cached_models=self.cached_models,
                return_models=True,  # Request models for caching
                progress_callback=self._on_progress_update,
                status_callback=self._on_status_update
            )

            # Extract loaded models for caching
            loaded_models = result.pop('_loaded_models', {})
            if loaded_models:
                # Emit loaded models signal for caching
                self.signals.models_loaded.emit(loaded_models)

            # Format results for display
            formatted_result = self._format_transcription_result(result, format_timestamp)
            result['formatted'] = formatted_result

            self.signals.progress_updated.emit(100)
            self.signals.status_updated.emit("Transcription completed")
            self.signals.transcription_completed.emit(result)
            self.signals.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def _format_transcription_result(self, result: Dict[str, Any], format_timestamp) -> Dict[str, str]:
        """
        Format transcription results for display using SRT timestamp format.
        Matches the original formatting logic from whisperx_bridge.py.
        """
        formatted = {
            'raw': '',
            'timestamped': '',
            'speakers': '',
            'full': ''
        }

        # Determine which result to use (priority: speakers > aligned > basic)
        segments = None
        if 'segments_with_speakers' in result:
            segments_data = result['segments_with_speakers']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                segments = segments_data['segments']
            else:
                segments = segments_data
        elif 'aligned' in result:
            segments_data = result['aligned']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                segments = segments_data['segments']
            else:
                segments = segments_data
        elif 'transcription' in result:
            segments_data = result['transcription']
            if isinstance(segments_data, dict) and 'segments' in segments_data:
                segments = segments_data['segments']
            else:
                segments = segments_data

        if segments:
            raw_text = []
            timestamped_text = []
            speaker_text = []
            full_text = []

            for segment in segments:
                text = segment.get('text', '').strip()
                if not text:
                    continue

                # Raw text
                raw_text.append(text)

                # Format timestamps in SRT format (HH:MM:SS,mmm)
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                start_srt = format_timestamp(start, always_include_hours=True, decimal_marker=',')
                end_srt = format_timestamp(end, always_include_hours=True, decimal_marker=',')

                # Timestamped text with SRT format
                timestamped_text.append(f"[{start_srt} --> {end_srt}] {text}")

                # Speaker text
                speaker = segment.get('speaker', 'Unknown')
                if speaker and speaker != 'Unknown':
                    speaker_text.append(f"{speaker}: {text}")
                else:
                    speaker_text.append(text)

                # Full text (timestamps + speakers) with SRT format
                if speaker and speaker != 'Unknown':
                    full_text.append(f"[{start_srt} --> {end_srt}] {speaker}: {text}")
                else:
                    full_text.append(f"[{start_srt} --> {end_srt}] {text}")

            formatted['raw'] = ' '.join(raw_text)
            formatted['timestamped'] = '\n'.join(timestamped_text)
            formatted['speakers'] = '\n'.join(speaker_text)
            formatted['full'] = '\n'.join(full_text)

        return formatted
