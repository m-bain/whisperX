"""
Tests for --diarize_device CLI flag.

Verifies that:
1. The CLI parses --diarize_device correctly.
2. transcribe_task passes diarize_device (not device) to DiarizationPipeline.
3. When --diarize_device is omitted, it falls back to --device.
"""
import argparse
import sys
import os
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure the local whisperx package takes precedence over any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whisperx.__main__ import cli
from whisperx.transcribe import transcribe_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> dict:
    """Return a complete args dict as produced by the CLI, with sensible defaults."""
    defaults = {
        # audio
        "audio": ["fake_audio.wav"],
        # model
        "model": "tiny",
        "model_cache_only": False,
        "model_dir": None,
        "output_dir": "/tmp",
        "output_format": "json",
        # devices
        "device": "cpu",
        "device_index": 0,
        "diarize_device": None,       # <-- the new flag
        # inference
        "batch_size": 8,
        "compute_type": "int8",
        "verbose": False,
        "log_level": None,
        "task": "transcribe",
        "language": "en",
        # alignment (disabled to keep tests fast)
        "align_model": None,
        "interpolate_method": "nearest",
        "no_align": True,
        "return_char_alignments": False,
        # vad
        "vad_method": "pyannote",
        "vad_onset": 0.500,
        "vad_offset": 0.363,
        "chunk_size": 30,
        # diarization
        "diarize": True,
        "min_speakers": None,
        "max_speakers": None,
        "diarize_model": "pyannote/speaker-diarization-community-1",
        "speaker_embeddings": False,
        "print_progress": False,
        # transcription options
        "temperature": 0,
        "temperature_increment_on_fallback": None,
        "best_of": 5,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "suppress_tokens": "-1",
        "suppress_numerals": False,
        "initial_prompt": None,
        "hotwords": None,
        "condition_on_previous_text": False,
        "fp16": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        # writer
        "highlight_words": False,
        "max_line_width": None,
        "max_line_count": None,
        "segment_resolution": "sentence",
        # misc
        "threads": 0,
        "hf_token": "hf_test_token",
    }
    defaults.update(overrides)
    return defaults


def _mock_transcribe_result():
    return {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "language": "en",
    }


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCliArgParsing:
    """Test that --diarize_device is wired up in the argument parser."""

    def _parse(self, argv):
        """Parse argv using the real CLI parser, return the args dict."""
        import importlib
        import whisperx.__main__ as main_module

        # Temporarily replace transcribe_task so cli() doesn't actually run
        with patch("whisperx.__main__.importlib.metadata.version", return_value="0.0.0"), \
             patch("whisperx.transcribe.transcribe_task"):
            # Build a parser the same way cli() does, but grab args before execution
            parser = argparse.ArgumentParser()
            # Re-use the same setup by invoking cli() with sys.argv mocked,
            # capturing the parsed namespace via a side-effect on transcribe_task.
            captured = {}

            def capture(args, _parser):
                captured.update(args)

            with patch("whisperx.__main__.importlib"), \
                 patch("whisperx.transcribe.transcribe_task", side_effect=capture):
                old_argv = sys.argv
                try:
                    sys.argv = ["whisperx"] + argv
                    try:
                        cli()
                    except SystemExit:
                        pass
                finally:
                    sys.argv = old_argv

            return captured

    def test_diarize_device_defaults_to_none(self):
        args = self._parse(["audio.wav"])
        assert "diarize_device" in args
        assert args["diarize_device"] is None

    def test_diarize_device_mps(self):
        args = self._parse(["audio.wav", "--diarize_device", "mps"])
        assert args["diarize_device"] == "mps"

    def test_diarize_device_cuda(self):
        args = self._parse(["audio.wav", "--diarize_device", "cuda"])
        assert args["diarize_device"] == "cuda"

    def test_diarize_device_cpu(self):
        args = self._parse(["audio.wav", "--diarize_device", "cpu"])
        assert args["diarize_device"] == "cpu"


# ---------------------------------------------------------------------------
# transcribe_task device routing tests
# ---------------------------------------------------------------------------

MOCK_TARGETS = {
    "load_model": "whisperx.transcribe.load_model",
    "load_audio": "whisperx.transcribe.load_audio",
    "load_align_model": "whisperx.transcribe.load_align_model",
    "align": "whisperx.transcribe.align",
    "DiarizationPipeline": "whisperx.transcribe.DiarizationPipeline",
    "assign_word_speakers": "whisperx.transcribe.assign_word_speakers",
    "get_writer": "whisperx.transcribe.get_writer",
}


class TestDiarizeDeviceRouting:
    """Test that DiarizationPipeline receives the correct device."""

    def _run(self, args: dict):
        """Run transcribe_task with all heavy dependencies mocked."""
        parser = argparse.ArgumentParser()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = _mock_transcribe_result()

        mock_diarize_instance = MagicMock()
        mock_diarize_instance.return_value = MagicMock()  # diarize_segments

        mock_writer = MagicMock()
        mock_writer.return_value = MagicMock()

        with patch(MOCK_TARGETS["load_model"], return_value=mock_model), \
             patch(MOCK_TARGETS["load_audio"], return_value=MagicMock()), \
             patch(MOCK_TARGETS["load_align_model"], return_value=(MagicMock(), MagicMock())), \
             patch(MOCK_TARGETS["align"], return_value=_mock_transcribe_result()), \
             patch(MOCK_TARGETS["DiarizationPipeline"], return_value=mock_diarize_instance) as MockDP, \
             patch(MOCK_TARGETS["assign_word_speakers"], return_value=_mock_transcribe_result()), \
             patch(MOCK_TARGETS["get_writer"], return_value=mock_writer):

            transcribe_task(args.copy(), parser)
            return MockDP

    def test_uses_diarize_device_when_specified(self):
        """DiarizationPipeline must use diarize_device, not device."""
        args = _make_args(device="cpu", diarize_device="mps")
        MockDP = self._run(args)
        MockDP.assert_called_once()
        _, kwargs = MockDP.call_args
        assert kwargs["device"] == "mps", (
            f"Expected device='mps', got device={kwargs['device']!r}"
        )

    def test_falls_back_to_device_when_diarize_device_is_none(self):
        """When diarize_device is None, DiarizationPipeline must use device."""
        args = _make_args(device="cpu", diarize_device=None)
        MockDP = self._run(args)
        MockDP.assert_called_once()
        _, kwargs = MockDP.call_args
        assert kwargs["device"] == "cpu", (
            f"Expected device='cpu', got device={kwargs['device']!r}"
        )

    def test_diarize_device_overrides_device(self):
        """diarize_device must override device even when both are non-default."""
        args = _make_args(device="cuda", diarize_device="cuda:1")
        MockDP = self._run(args)
        _, kwargs = MockDP.call_args
        assert kwargs["device"] == "cuda:1"

    def test_device_unchanged_for_transcription(self):
        """Changing diarize_device must not affect the device passed to load_model."""
        args = _make_args(device="cpu", diarize_device="mps")
        parser = argparse.ArgumentParser()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = _mock_transcribe_result()

        with patch(MOCK_TARGETS["load_model"], return_value=mock_model) as MockLM, \
             patch(MOCK_TARGETS["load_audio"], return_value=MagicMock()), \
             patch(MOCK_TARGETS["load_align_model"], return_value=(MagicMock(), MagicMock())), \
             patch(MOCK_TARGETS["align"], return_value=_mock_transcribe_result()), \
             patch(MOCK_TARGETS["DiarizationPipeline"], return_value=MagicMock(return_value=MagicMock())), \
             patch(MOCK_TARGETS["assign_word_speakers"], return_value=_mock_transcribe_result()), \
             patch(MOCK_TARGETS["get_writer"], return_value=MagicMock(return_value=MagicMock())):

            transcribe_task(args.copy(), parser)
            _, kwargs = MockLM.call_args
            assert kwargs["device"] == "cpu", (
                "diarize_device must not affect the transcription device"
            )
