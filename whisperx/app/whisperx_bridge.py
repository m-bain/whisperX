"""
Bridge between Qt application and WhisperX functionality.
Adapts WhisperX modules for use in Qt threading environment.
"""
import time
import os
from typing import Dict, Any, Optional, Callable
import torch

# WhisperX imports
from whisperx.asr import load_model
from whisperx.alignment import load_align_model, align
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.types import TranscriptionResult
from whisperx.utils import format_timestamp
from whisperx.app.app_config import TranscriptionConfig

class WhisperXBridge:
    def __init__(self):
        self.loaded_models = {}
        self._current_progress = 0

    def load_models(self, config: TranscriptionConfig,
                    progress_callback: Optional[Callable] = None,
                    status_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """ Load all required models"""

        models = {}
        total_steps = 3 # ASR, alignment, diarization
        current_step = 0

        def update_progress(step_progress: float):
            """Update overall progress."""
            overall_progress = int(((current_step + step_progress) / total_steps) * 100)
            if progress_callback:
                progress_callback(overall_progress)

        try:
            # Load ASR model
            if status_callback:
                status_callback("Loading speech recognition model...")
            update_progress(0)

            whisperx_params = config.to_whisperx_params()
            asr_model = load_model(
                whisper_arch=whisperx_params['model_name'],
                device=whisperx_params['device'],
                device_index=whisperx_params['device_index'],
                compute_type=whisperx_params['compute_type'],
                language=whisperx_params['language'],
                # vad_method=whisperx_params['vad_method'],
                # vad_options={
                #     'vad_onset': whisperx_params['vad_onset'],
                #     'vad_offset': whisperx_params['vad_offset']
                # }
            )
            models['asr'] = asr_model
            current_step += 1
            update_progress(0)

            # Load alignment model if needed
            if config.enable_alignment:
                if status_callback:
                    status_callback("Loading alignment model...")

                # Determine language for alignment model
                language = config.language or "en"

                align_model, align_metadata = load_align_model(
                    language_code=language,
                    device=whisperx_params['device']
                )
                models['alignment'] = {
                    'model': align_model,
                    'metadata': align_metadata
                }
            current_step += 1
            update_progress(0)

            # Load diarization model if needed
            if config.enable_diarization:
                if status_callback:
                    status_callback("Loading speaker diarization model...")

                diarization_pipeline = DiarizationPipeline(
                    use_auth_token=None,  # You may need to handle HF tokens
                    device=torch.device(whisperx_params['device'])
                )
                models['diarization'] = diarization_pipeline

            current_step += 1
            update_progress(0)

            self.loaded_models = models
            return models

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def transcribe_audio(self, config: TranscriptionConfig,
                         models: Optional[Dict] = None,
                         progress_callback: Optional[Callable] = None,
                         status_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform complete transcription pipeline with real progress tracking."""

        if models is None:
            models = self.loaded_models

        if not models or 'asr' not in models:
            raise RuntimeError("ASR model not loaded")

        result = {}

        # Progress phases: Audio(5%) + Transcription(55%) + Alignment(25%) + Diarization(15%)
        phase_weights = {
            'audio': 5,
            'transcription': 55,
            'alignment': 25,
            'diarization': 15
        }

        current_phase_start = 0

        def phase_progress_callback(phase: str, progress: int):
            """Convert phase-specific progress to overall progress."""
            if progress_callback:
                phase_weight = phase_weights[phase]
                overall_progress = current_phase_start + int((progress * phase_weight) / 100)
                progress_callback(min(overall_progress, 100))

        def phase_status_callback(message: str):
            """Pass through status messages."""
            if status_callback:
                status_callback(message)

        try:
            # Phase 1: Load audio
            if status_callback:
                status_callback("Loading audio file...")
            if progress_callback:
                progress_callback(0)

            start = time.time()
            audio = load_audio(config.audio_file)
            end = time.time()
            current_phase_start += phase_weights['audio']
            print(f"Audio loading took {end - start:.2f} seconds")

            # Phase 2: Transcribe with real progress
            if status_callback:
                status_callback("Performing speech recognition...")

            asr_model = models['asr']
            start = time.time()
            transcribe_result = asr_model.transcribe(
                audio=audio,
                batch_size=config.batch_size,
                # chunk_size=config.chunk_size,
                print_progress=False,  # Disable print, use callback
                combined_progress=False,  # We handle combination ourselves
                progress_callback=None, #lambda p: phase_progress_callback('transcription', p),
                status_callback=None #phase_status_callback
            )
            end = time.time()
            print(f"Transcription took {end - start:.2f} seconds")
            result['transcription'] = transcribe_result
            current_phase_start += phase_weights['transcription']

            # Phase 3: Alignment with real progress
            if config.enable_alignment and 'alignment' in models:
                if status_callback:
                    status_callback("Aligning timestamps...")

                align_model = models['alignment']['model']
                align_metadata = models['alignment']['metadata']
                start = time.time()
                aligned_result = align(
                    transcript=transcribe_result["segments"],
                    model=align_model,
                    align_model_metadata=align_metadata,
                    audio=audio,
                    device=config.device,
                    print_progress=False,  # Disable print, use callback
                    combined_progress=False,  # We handle combination ourselves
                    progress_callback=lambda p: phase_progress_callback('alignment', p),
                    status_callback=phase_status_callback
                )
                end = time.time()
                result['aligned_transcription'] = aligned_result
                print(f"Alignment took {end - start:.2f} seconds")

            current_phase_start += phase_weights['alignment']

            # Phase 4: Diarization with manual progress tracking
            if config.enable_diarization and 'diarization' in models:
                if status_callback:
                    status_callback("Identifying speakers...")

                # Diarization progress (manual since pyannote doesn't expose progress)
                phase_progress_callback('diarization', 20)
                start = time.time()
                diarization_pipeline = models['diarization']
                diarization_result = diarization_pipeline(config.audio_file)

                phase_progress_callback('diarization', 60)

                # Assign speakers to words
                segments_with_speakers = assign_word_speakers(
                    diarization_result,
                    result.get('aligned_transcription', transcribe_result)
                )
                end = time.time()
                print(f"Diarization took {end - start:.2f} seconds")
                result['diarization'] = diarization_result
                result['segments_with_speakers'] = segments_with_speakers

                phase_progress_callback('diarization', 100)

            # Final formatting
            if status_callback:
                status_callback("Formatting results...")

            formatted_result = self._format_transcription_result(result, config)
            result['formatted'] = formatted_result

            if progress_callback:
                progress_callback(100)

            return result

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def _format_transcription_result(self, result: Dict[str, Any],
                                     config: TranscriptionConfig) -> Dict[str, str]:
        """Format transcription results for display using SRT timestamp format."""
        formatted = {
            'raw': '',
            'timestamped': '',
            'speakers': '',
            'full': ''
        }

        # Determine which result to use
        segments = None
        if 'segments_with_speakers' in result:
            segments = result['segments_with_speakers']['segments']
        elif 'aligned_transcription' in result:
            segments = result['aligned_transcription']['segments']
        elif 'transcription' in result:
            segments = result['transcription']['segments']

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

