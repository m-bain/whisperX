"""Test that align() produces word-level timestamps for unalignable characters."""

import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

from whisperx.alignment import align
from whisperx.utils import interpolate_nans


def _make_mock_model(emission, dictionary):
    """Create a mock torchaudio-style model that returns a fixed emission matrix.

    The emission should be pre-log-softmax logits of shape (num_frames, vocab_size).
    align() will apply log_softmax itself.
    """
    model = MagicMock()
    # torchaudio interface: model(waveform, lengths=lengths) -> (emissions, _)
    # emissions shape: (batch=1, num_frames, vocab_size)
    model.return_value = (emission.unsqueeze(0), None)
    return model


def _make_emission(num_frames, dictionary, transcript_chars, blank_id=0):
    """Build a synthetic emission matrix where known chars peak at the right frames.

    Distributes characters evenly across frames. Known chars get high logits
    at their assigned frames. Unknown chars have no specific token but will
    get wildcard treatment in align().
    """
    vocab_size = max(dictionary.values()) + 1
    # Start with uniform low logits, blank slightly favored
    emission = torch.full((num_frames, vocab_size), -5.0)
    emission[:, blank_id] = -1.0

    # Assign each transcript char a span of frames
    chars_in_dict = [(i, c) for i, c in enumerate(transcript_chars)
                     if c.lower() in dictionary]
    if chars_in_dict:
        frames_per_char = num_frames // (len(transcript_chars) + 1)
        for seq_idx, (char_idx, char) in enumerate(chars_in_dict):
            center = (char_idx + 1) * frames_per_char
            start = max(0, center - frames_per_char // 2)
            end = min(num_frames, center + frames_per_char // 2)
            token_id = dictionary[char.lower()]
            for t in range(start, end):
                emission[t, token_id] = 2.0  # high logit for this token
                emission[t, blank_id] = -3.0  # suppress blank

    return emission


class TestAlignWithWildcards:
    """Test align() end-to-end with unknown characters."""

    DICTIONARY = {
        "<pad>": 0,  # blank
        "a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
        "f": 6, "g": 7, "h": 8, "i": 9, "k": 10,
        "l": 11, "m": 12, "n": 13, "o": 14, "p": 15,
        "r": 16, "s": 17, "t": 18, "u": 19, "w": 20,
        "|": 21,
    }
    METADATA = {"language": "en", "dictionary": DICTIONARY, "type": "torchaudio"}

    def _run_align(self, text, duration=5.0, num_frames=100, interpolate_method="nearest"):
        """Run align() with a mock model on a single segment."""
        torch.manual_seed(0)
        emission = _make_emission(num_frames, self.DICTIONARY, list(text), blank_id=0)
        model = _make_mock_model(emission, self.DICTIONARY)

        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        audio = torch.randn(num_samples)

        transcript = [{"text": text, "start": 0.0, "end": duration}]
        result = align(
            transcript=transcript,
            model=model,
            align_model_metadata=self.METADATA,
            audio=audio,
            device="cpu",
            interpolate_method=interpolate_method,
        )
        return result

    def test_known_chars_get_timestamps(self):
        """Baseline: words with all known chars get timestamps."""
        result = self._run_align("the cat sat")
        for word in result["word_segments"]:
            assert "start" in word, f"'{word['word']}' missing start"
            assert "end" in word, f"'{word['word']}' missing end"
            assert "score" in word, f"'{word['word']}' missing score"

    def test_unknown_word_gets_timestamps(self):
        """A word made of unknown chars (digits) gets timestamps via wildcard."""
        result = self._run_align("cost 43 dollars")
        words = {w["word"]: w for w in result["word_segments"]}
        assert "43" in words, f"'43' not in word_segments: {list(words.keys())}"
        assert "start" in words["43"], "'43' missing start timestamp"
        assert "end" in words["43"], "'43' missing end timestamp"
        assert "score" in words["43"], "'43' missing score"

    def test_mixed_word_gets_timestamps(self):
        """A word with mixed known/unknown chars gets timestamps."""
        result = self._run_align("has 43k users")
        # "43k" has unknown '4','3' and known 'k'
        words = {w["word"]: w for w in result["word_segments"]}
        assert "43k" in words, f"'43k' not in word_segments: {list(words.keys())}"
        assert "start" in words["43k"]
        assert "end" in words["43k"]

    def test_unknown_word_does_not_corrupt_neighbors(self):
        """Known words adjacent to unknown words still get valid timestamps."""
        result = self._run_align("cost 43 dollars")
        words = {w["word"]: w for w in result["word_segments"]}
        for known in ("cost", "dollars"):
            assert known in words
            assert "start" in words[known], f"'{known}' missing start"
            assert "end" in words[known], f"'{known}' missing end"
            assert "score" in words[known], f"'{known}' missing score"

    def test_all_unknown_segment_gets_timestamps(self):
        """A segment with only unknown chars gets wildcard-aligned timestamps."""
        result = self._run_align("123 456")
        assert len(result["word_segments"]) > 0, "expected word_segments for all-unknown text"
        for word in result["word_segments"]:
            assert "start" in word, f"'{word['word']}' missing start"
            assert "end" in word, f"'{word['word']}' missing end"

    def test_timestamps_are_ordered(self):
        """Word timestamps are monotonically non-decreasing."""
        result = self._run_align("the 99 cats")
        starts = [w["start"] for w in result["word_segments"] if "start" in w]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], (
                f"Timestamps not ordered: {starts}"
            )

    def test_issue_1372_digits_comma_no_timestamps(self):
        """Regression: '4,9' (digits+comma) must get timestamps.

        https://github.com/m-bain/whisperX/issues/1372#issuecomment-4051234966
        Reporter showed that align() returned {'word': '4,9'} with no
        start/end/score for German text containing '4,9'.
        """
        result = self._run_align("halt mit 4,9 nicht ins parlament", num_frames=200)
        words = {w["word"]: w for w in result["word_segments"]}
        assert "4,9" in words, f"'4,9' not in word_segments: {list(words.keys())}"
        assert "start" in words["4,9"], "'4,9' missing start"
        assert "end" in words["4,9"], "'4,9' missing end"
        assert "score" in words["4,9"], "'4,9' missing score"

    def test_ignore_does_not_crash(self):
        """interpolate_method='ignore' must not crash (issue #1368)."""
        result = self._run_align("the 99 cats", interpolate_method="ignore")
        words = {w["word"]: w for w in result["word_segments"]}
        assert len(words) == 3
        for known in ("the", "cats"):
            assert "start" in words[known], f"'{known}' missing start"
            assert "end" in words[known], f"'{known}' missing end"

    def test_ignore_segments_have_valid_timestamps(self):
        """Segments always have valid float start/end even with interpolate_method='ignore'."""
        result = self._run_align("the 99 cats", interpolate_method="ignore")
        for seg in result["segments"]:
            assert isinstance(seg["start"], float), f"segment start not float: {seg['start']}"
            assert isinstance(seg["end"], float), f"segment end not float: {seg['end']}"
            assert not np.isnan(seg["start"]), "segment start is NaN"
            assert not np.isnan(seg["end"]), "segment end is NaN"


class TestInterpolateNans:
    """Unit tests for interpolate_nans()."""

    def test_ignore_preserves_nans(self):
        x = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = interpolate_nans(x, method="ignore")
        assert result is x
        assert result.isna().sum() == 2

    def test_nearest_fills_nans(self):
        x = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = interpolate_nans(x, method="nearest")
        assert result.notna().all()

    def test_ignore_word_assignment_integration(self):
        """Simulate the word-timestamp assignment logic from alignment.py:365-376.

        With wildcard CTC, this path is rarely reached in practice, but it's
        the code that "ignore" is designed to affect. This test proves the
        pieces fit together: interpolate_nans preserves NaN, and the pd.notna
        guard prevents timestamp assignment to unaligned words.
        """
        sentence_words = [
            {"word": "the", "start": 0.1, "end": 0.3},
            {"word": "99"},  # simulates a word with no CTC timestamps
            {"word": "cats", "start": 0.6, "end": 0.9},
        ]
        _starts = pd.Series([0.1, np.nan, 0.6])
        _ends = pd.Series([0.3, np.nan, 0.9])

        # With "ignore": NaNs preserved, unaligned word stays without timestamps
        _starts_ign = interpolate_nans(_starts, method="ignore")
        _ends_ign = interpolate_nans(_ends, method="ignore")
        for i, w in enumerate(sentence_words):
            if "start" not in w and pd.notna(_starts_ign.iloc[i]):
                w["start"] = _starts_ign.iloc[i]
            if "end" not in w and pd.notna(_ends_ign.iloc[i]):
                w["end"] = _ends_ign.iloc[i]

        assert "start" not in sentence_words[1], "'99' should not get start with ignore"
        assert "end" not in sentence_words[1], "'99' should not get end with ignore"

        # With "nearest": NaNs filled, unaligned word gets interpolated timestamps
        sentence_words_nearest = [
            {"word": "the", "start": 0.1, "end": 0.3},
            {"word": "99"},
            {"word": "cats", "start": 0.6, "end": 0.9},
        ]
        _starts_nr = interpolate_nans(_starts.copy(), method="nearest")
        _ends_nr = interpolate_nans(_ends.copy(), method="nearest")
        for i, w in enumerate(sentence_words_nearest):
            if "start" not in w and pd.notna(_starts_nr.iloc[i]):
                w["start"] = _starts_nr.iloc[i]
            if "end" not in w and pd.notna(_ends_nr.iloc[i]):
                w["end"] = _ends_nr.iloc[i]

        assert "start" in sentence_words_nearest[1], "'99' should get start with nearest"
        assert "end" in sentence_words_nearest[1], "'99' should get end with nearest"
