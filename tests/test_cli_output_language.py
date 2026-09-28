"""Test that the CLI reports the transcribed language rather than the --language default."""

import json
import sys
from unittest.mock import patch

import numpy as np

import whisperx.transcribe as transcribe_module
from whisperx.__main__ import cli

SEGMENTS = [{"start": 0.0, "end": 2.0, "text": "今日はいい天気"}]
WORDS = [
    {"word": "今日", "start": 0.0, "end": 0.6, "score": 0.9},
    {"word": "は", "start": 0.6, "end": 0.9, "score": 0.9},
    {"word": "いい天気", "start": 0.9, "end": 2.0, "score": 0.9},
]


class StubModel:
    """Stands in for the faster-whisper pipeline and reports a fixed language."""

    def __init__(self, language):
        self.language = language

    def transcribe(self, *args, **kwargs):
        return {"segments": [dict(seg) for seg in SEGMENTS], "language": self.language}


def stub_align(transcript, *args, **kwargs):
    """Mimic align(), which returns no language key."""
    return {
        "segments": [dict(transcript[0], words=[dict(word) for word in WORDS])],
        "word_segments": [dict(word) for word in WORDS],
    }


def run_cli(tmp_path, extra_args, detected="ja"):
    """Run the real CLI over one stub audio file and return its JSON and SRT output."""
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"\0")
    argv = ["whisperx", str(audio), "--output_dir", str(tmp_path), "--verbose", "False"]

    with (
        patch.object(transcribe_module, "load_model", lambda *a, **k: StubModel(detected)),
        patch.object(transcribe_module, "load_audio", lambda *a, **k: np.zeros(16000, dtype=np.float32)),
        patch.object(transcribe_module, "load_align_model", lambda *a, **k: (object(), {"language": detected})),
        patch.object(transcribe_module, "align", stub_align),
        patch.object(sys, "argv", argv + extra_args),
    ):
        cli()

    result = json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))
    srt = audio.with_suffix(".srt").read_text(encoding="utf-8")
    return result, srt


class TestCLIOutputLanguage:
    def test_detected_language_is_written(self, tmp_path):
        result, _ = run_cli(tmp_path, [])
        assert result["language"] == "ja"

    def test_detected_ja_joins_words_without_spaces(self, tmp_path):
        """Regression for #248: auto-detected ja must not gain spaces between words."""
        _, srt = run_cli(tmp_path, [])
        assert srt.strip().splitlines()[-1] == "今日はいい天気"

    def test_detected_language_is_written_without_alignment(self, tmp_path):
        result, _ = run_cli(tmp_path, ["--no_align"])
        assert result["language"] == "ja"

    def test_explicit_language_is_written(self, tmp_path):
        result, _ = run_cli(tmp_path, ["--language", "ja"])
        assert result["language"] == "ja"

    def test_english_keeps_spaces(self, tmp_path):
        _, srt = run_cli(tmp_path, [], detected="en")
        assert srt.strip().splitlines()[-1] == "今日 は いい天気"
