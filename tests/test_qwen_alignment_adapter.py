from types import SimpleNamespace

import numpy as np
import pytest

from whisperx.alignment_qwen import align
from whisperx.audio import SAMPLE_RATE


class DummyQwenAligner:
    def align(self, audio, text, language):
        assert len(audio) == 1
        assert len(text) == 1
        assert language == ["English"]
        return [
            [
                SimpleNamespace(text="hello", start_time=0.100, end_time=0.450),
                SimpleNamespace(text="world", start_time=0.450, end_time=0.900),
            ]
        ]


def test_qwen_align_offsets_segment_timestamps_and_builds_word_segments():
    transcript = [{"start": 1.0, "end": 2.0, "text": "hello world"}]
    audio = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
    metadata = {"language": "en", "type": "qwen_forced_aligner"}

    out = align(
        transcript=transcript,
        model=DummyQwenAligner(),
        align_model_metadata=metadata,
        audio=audio,
        device="cpu",
    )

    assert len(out["segments"]) == 1
    assert len(out["segments"][0]["words"]) == 2
    assert out["segments"][0]["words"][0]["word"] == "hello"
    assert out["segments"][0]["words"][0]["start"] == 1.1
    assert out["segments"][0]["words"][0]["end"] == 1.45
    assert out["segments"][0]["words"][1]["word"] == "world"
    assert out["word_segments"][0]["word"] == "hello"
    assert out["word_segments"][1]["word"] == "world"


def test_qwen_align_rejects_char_level_alignment_flag():
    transcript = [{"start": 0.0, "end": 1.0, "text": "abc"}]
    audio = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    metadata = {"language": "en", "type": "qwen_forced_aligner"}

    with pytest.raises(NotImplementedError):
        align(
            transcript=transcript,
            model=DummyQwenAligner(),
            align_model_metadata=metadata,
            audio=audio,
            device="cpu",
            return_char_alignments=True,
        )
