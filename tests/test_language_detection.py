"""Tests for top-k language detection probabilities (no models are loaded)."""

import warnings
from unittest import mock

import numpy as np
import pytest

from whisperx import asr
from whisperx.audio import N_SAMPLES
from whisperx.vads import Vad

# ctranslate2's detect_language returns, per batch item, (token, probability)
# pairs sorted from the most to the least likely language.
DETECTED = [[("<|de|>", 0.7), ("<|en|>", 0.2), ("<|nl|>", 0.06), ("<|fr|>", 0.04)]]


class FakeVad(Vad):
    def __init__(self):
        super().__init__(vad_onset=0.5)

    def __call__(self, inputs, **kwargs):
        return None

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size, onset, offset):
        return [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]


def make_pipeline(language=None):
    model = mock.MagicMock()
    model.feat_kwargs = {"feature_size": 80}
    model.model.detect_language.return_value = DETECTED
    return asr.FasterWhisperPipeline(
        model=model,
        vad=FakeVad(),
        vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
        options=mock.MagicMock(),
        device="cpu",
        language=language,
    )


def transcribe(pipeline, **kwargs):
    outputs = [{"text": ["hallo"], "avg_logprob": [-0.1]}]
    with mock.patch.object(asr, "Tokenizer") as tokenizer_cls, mock.patch.object(
        asr.FasterWhisperPipeline, "__call__", side_effect=lambda *a, **k: iter(outputs)
    ), warnings.catch_warnings():
        # a single fake VAD chunk triggers the "batch_size >= total VAD chunks" warning
        warnings.simplefilter("ignore", UserWarning)
        result = pipeline.transcribe(np.zeros(N_SAMPLES, dtype=np.float32), batch_size=1, **kwargs)
    return result, tokenizer_cls


def test_detect_language_returns_most_likely_language():
    pipeline = make_pipeline()
    assert pipeline.detect_language(np.zeros(N_SAMPLES, dtype=np.float32)) == "de"


def test_detect_language_probabilities_all_and_top_k():
    pipeline = make_pipeline()
    audio = np.zeros(N_SAMPLES, dtype=np.float32)

    probs = pipeline.detect_language_probabilities(audio)
    assert probs == {"de": 0.7, "en": 0.2, "nl": 0.06, "fr": 0.04}
    assert list(probs) == ["de", "en", "nl", "fr"]

    assert list(pipeline.detect_language_probabilities(audio, top_k=2).items()) == [("de", 0.7), ("en", 0.2)]
    # top_k larger than the number of languages returns all of them
    assert len(pipeline.detect_language_probabilities(audio, top_k=100)) == 4


@pytest.mark.parametrize("top_k", [0, -1])
def test_detect_language_probabilities_rejects_non_positive_top_k(top_k):
    pipeline = make_pipeline()
    with pytest.raises(ValueError):
        pipeline.detect_language_probabilities(np.zeros(N_SAMPLES, dtype=np.float32), top_k=top_k)


def test_transcribe_returns_top_k_language_probs():
    pipeline = make_pipeline()
    result, tokenizer_cls = transcribe(pipeline, language_top_k=3)

    assert result["language"] == "de"
    assert list(result["language_probs"].items()) == [("de", 0.7), ("en", 0.2), ("nl", 0.06)]
    assert result["segments"][0]["text"] == "hallo"
    assert tokenizer_cls.call_args.kwargs["language"] == "de"
    # language detection runs only once
    assert pipeline.model.model.detect_language.call_count == 1


def test_transcribe_omits_language_probs_by_default():
    pipeline = make_pipeline()
    result, _ = transcribe(pipeline)

    assert result["language"] == "de"
    assert "language_probs" not in result


def test_transcribe_omits_language_probs_when_language_is_given():
    pipeline = make_pipeline()
    result, _ = transcribe(pipeline, language="fr", language_top_k=3)

    assert result["language"] == "fr"
    assert "language_probs" not in result
    pipeline.model.model.detect_language.assert_not_called()


def test_transcribe_rejects_non_positive_language_top_k():
    pipeline = make_pipeline()
    with pytest.raises(ValueError):
        transcribe(pipeline, language_top_k=0)
