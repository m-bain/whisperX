"""Test the multilingual MMS fallback aligner (align_mms).

These tests exercise the MMS-specific code path with a mock model and a
mock romanizer, so they run offline without downloading the MMS checkpoint
or loading the uroman data tables.
"""

import numpy as np
import torch
from unittest.mock import MagicMock, patch

from whisperx import alignment
from whisperx.alignment import align


DICTIONARY = {"<blank>": 0}
for _i, _c in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    DICTIONARY[_c] = _i


def _identity_uroman():
    """A romanizer that returns its input unchanged (already-Latin text)."""
    ur = MagicMock()
    ur.romanize_string.side_effect = lambda s, lcode=None: s
    return ur


def _make_mock_model(emission):
    """Mock a huggingface CTC model: model(waveform).logits -> (1, frames, vocab)."""
    model = MagicMock()
    model.return_value.logits = emission.unsqueeze(0)
    return model


def _make_emission(num_frames, rom_chars, blank_id=0):
    """Synthetic emission where each known romanized char peaks at its frame."""
    vocab_size = max(DICTIONARY.values()) + 1
    emission = torch.full((num_frames, vocab_size), -5.0)
    emission[:, blank_id] = -1.0

    known = [(i, c) for i, c in enumerate(rom_chars) if c in DICTIONARY]
    if known:
        frames_per_char = num_frames // (len(rom_chars) + 1)
        for char_idx, char in known:
            center = (char_idx + 1) * frames_per_char
            start = max(0, center - frames_per_char // 2)
            end = min(num_frames, center + frames_per_char // 2)
            token_id = DICTIONARY[char]
            for t in range(start, end):
                emission[t, token_id] = 2.0
                emission[t, blank_id] = -3.0
    return emission


def _rom_chars(text, use_chars):
    units = [c for c in text if not c.isspace()] if use_chars else text.split()
    return [ch for unit in units for ch in unit.lower()]


def _run_align_mms(text, language="en", iso="eng", duration=5.0, num_frames=100,
                   interpolate_method="nearest"):
    use_chars = language in alignment.LANGUAGES_WITHOUT_SPACES
    emission = _make_emission(num_frames, _rom_chars(text, use_chars))
    model = _make_mock_model(emission)
    metadata = {
        "language": language,
        "dictionary": DICTIONARY,
        "type": "mms",
        "blank_id": 0,
        "iso": iso,
    }
    audio = torch.randn(int(duration * 16000))
    transcript = [{"text": text, "start": 0.0, "end": duration}]
    with patch.object(alignment, "_get_uroman", _identity_uroman):
        return align(
            transcript=transcript,
            model=model,
            align_model_metadata=metadata,
            audio=audio,
            device="cpu",
            interpolate_method=interpolate_method,
        )


class TestAlignMMS:
    def test_dispatches_to_mms(self):
        """align() routes type='mms' metadata to the MMS aligner."""
        result = _run_align_mms("the cat sat")
        assert "segments" in result and "word_segments" in result
        assert len(result["word_segments"]) == 3

    def test_known_words_get_timestamps(self):
        result = _run_align_mms("the cat sat")
        for word in result["word_segments"]:
            assert "start" in word, f"'{word['word']}' missing start"
            assert "end" in word, f"'{word['word']}' missing end"
            assert "score" in word, f"'{word['word']}' missing score"

    def test_unknown_unit_gets_timestamps_via_wildcard(self):
        """A unit with chars outside the vocab still aligns through the <star> column."""
        result = _run_align_mms("the 9 cat")
        words = {w["word"]: w for w in result["word_segments"]}
        assert "9" in words, f"'9' not in word_segments: {list(words.keys())}"
        assert "start" in words["9"], "'9' missing start"
        assert "end" in words["9"], "'9' missing end"

    def test_timestamps_are_ordered(self):
        result = _run_align_mms("the cat sat on")
        starts = [w["start"] for w in result["word_segments"] if "start" in w]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], f"timestamps not ordered: {starts}"

    def test_spaceless_language_aligns_per_char(self):
        """For languages without spaces, each character becomes its own unit."""
        result = _run_align_mms("abc", language="ja", iso="jpn")
        assert [w["word"] for w in result["word_segments"]] == ["a", "b", "c"]

    def test_segment_bounds_are_valid_floats(self):
        result = _run_align_mms("the cat sat")
        for seg in result["segments"]:
            assert isinstance(seg["start"], float) and not np.isnan(seg["start"])
            assert isinstance(seg["end"], float) and not np.isnan(seg["end"])
            assert seg["end"] >= seg["start"]
