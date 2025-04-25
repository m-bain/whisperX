"""
WhisperX: Enhanced Automatic Speech Recognition with Word-Level Timestamps

WhisperX extends OpenAI's Whisper model with:
- Faster transcription via faster-whisper
- Word-level timestamp alignment using phoneme-based forced alignment
- Speaker diarization capabilities
- Improved voice activity detection (VAD)
- Flexible output formats (SRT, VTT, JSON, etc.)

The package builds on Whisper's strong ASR capabilities and adds precise word-level
timestamps by aligning the transcript to a fine-tuned phoneme-level ASR model.
"""

# Core ASR functionality
from whisperx.asr import load_model
from whisperx.asr_openai_whisper import load_model as load_openai_whisper_model

# Audio utilities
from whisperx.audio import load_audio

# Alignment functionality for word-level timestamps
from whisperx.alignment import load_align_model, align

# Speaker diarization
from whisperx.diarize import (
    assign_word_speakers,
    DiarizationPipeline,
)