"""Tests for handling empty segments and silence during alignment without raising KeyError."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


def test_empty_aligned_subsegments_handling():
    """Verify that an empty DataFrame or DataFrame without 'start' column resolves to an empty list without KeyError."""
    # Test case 1: Completely empty DataFrame (simulating aligned_subsegments = [])
    empty_subsegments = []
    df = pd.DataFrame(empty_subsegments)

    if not df.empty and "start" in df:
        result = df.to_dict("records")
    else:
        result = []

    assert result == []

    # Test case 2: DataFrame missing 'start' column
    df_no_start = pd.DataFrame([{"text": "some text", "words": []}])
    if not df_no_start.empty and "start" in df_no_start:
        result = df_no_start.to_dict("records")
    else:
        result = []

    assert result == []


def test_align_with_empty_sentence_spans():
    """Mock-based test verifying align() handles segments that produce no aligned subsegments without KeyError."""
    import torch
    from whisperx.alignment import align

    mock_model = MagicMock()
    # Mock torchaudio output: (batch=1, num_frames=50, vocab_size=5)
    emission = torch.randn(1, 50, 5)
    mock_model.return_value = (emission, None)

    metadata = {
        "language": "en",
        "dictionary": {"<pad>": 0, "a": 1, "b": 2, "|": 3},
        "type": "torchaudio",
    }
    sample_rate = 16000
    audio = torch.zeros(sample_rate)

    # Scenario: sentence splitter returns no spans, leading to empty aligned_subsegments
    transcript = [{"text": "a", "start": 0.0, "end": 1.0}]

    with patch("whisperx.alignment.SentenceTokenizer") as mock_splitter_cls:
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.span_tokenize.return_value = []
        mock_splitter_cls.return_value = mock_splitter_instance

        result = align(
            transcript=transcript,
            model=mock_model,
            align_model_metadata=metadata,
            audio=audio,
            device="cpu",
        )

        assert isinstance(result, dict)
        assert "segments" in result
        assert "word_segments" in result
        assert result["segments"] == []
        assert result["word_segments"] == []


def test_align_with_empty_transcript():
    """Verify align() handles empty transcript gracefully."""
    import torch
    from whisperx.alignment import align

    mock_model = MagicMock()
    metadata = {
        "language": "en",
        "dictionary": {"<pad>": 0, "a": 1, "|": 2},
        "type": "torchaudio",
    }
    audio = torch.zeros(16000)

    result = align(
        transcript=[],
        model=mock_model,
        align_model_metadata=metadata,
        audio=audio,
        device="cpu",
    )

    assert result == {"segments": [], "word_segments": []}
