"""Test that loading a VAD leaves the process-wide torch thread count alone."""

from unittest.mock import patch

import torch

from whisperx.vads.silero import Silero


def _fake_hub_load(*args, **kwargs):
    """Stand in for silero-vad, which pins torch to one thread when imported.

    snakers4/silero-vad, src/silero_vad/model.py line 3, calls
    torch.set_num_threads(1) at module scope, and torch.hub.load imports that
    module. Faked here so the test needs no network and no model download.
    """
    torch.set_num_threads(1)
    return object(), (None, None, None, None, None)


def test_silero_load_preserves_thread_count():
    """--threads must survive VAD loading: alignment and diarization run after it."""
    original = torch.get_num_threads()
    requested = 2
    try:
        torch.set_num_threads(requested)

        with patch("torch.hub.load", _fake_hub_load):
            Silero(vad_onset=0.500, vad_offset=0.363, chunk_size=30)

        assert torch.get_num_threads() == requested
    finally:
        torch.set_num_threads(original)
