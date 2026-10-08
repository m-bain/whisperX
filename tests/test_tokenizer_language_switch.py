"""transcribe() on a model loaded with a language, called with another one."""

import warnings

import numpy as np
import pytest
from faster_whisper.tokenizer import Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel

from whisperx.asr import FasterWhisperPipeline
from whisperx.vads import Vad

SPECIAL_TOKENS = ["<|transcribe|>", "<|translate|>", "<|en|>", "<|de|>", "<|pl|>"]


class _NoSpeechVad(Vad):
    def __init__(self):
        super().__init__(0.5)

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size, onset, offset):
        return []

    def __call__(self, inputs):
        return []


class _Model:
    def __init__(self, multilingual=True):
        vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS + ["[UNK]"])}
        self.hf_tokenizer = HFTokenizer(WordLevel(vocab, unk_token="[UNK]"))
        self.model = type("CT2Model", (), {"is_multilingual": multilingual})()


def _pipeline(language="pl", task="transcribe", multilingual=True):
    model = _Model(multilingual)
    return FasterWhisperPipeline(
        model=model,
        vad=_NoSpeechVad(),
        vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
        options=None,
        tokenizer=Tokenizer(model.hf_tokenizer, multilingual, task=task, language=language),
        language=language,
    )


def _transcribe(pipeline, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # batch_size >= number of VAD chunks
        return pipeline.transcribe(np.zeros(16000, dtype=np.float32), **kwargs)


@pytest.mark.parametrize("language", ["en", "de"])
def test_another_language_than_the_preset_one(language):
    pipeline = _pipeline(language="pl")

    result = _transcribe(pipeline, language=language)

    assert result["language"] == language
    assert pipeline.tokenizer.language_code == language
    assert pipeline.tokenizer.task == pipeline.tokenizer.transcribe


def test_switching_back_and_forth():
    pipeline = _pipeline(language="pl")

    for language in ["en", "pl", "en"]:
        assert _transcribe(pipeline, language=language)["language"] == language
        assert pipeline.tokenizer.language_code == language


def test_a_call_without_language_uses_the_preset_one_again():
    pipeline = _pipeline(language="pl")

    _transcribe(pipeline, language="en")
    result = _transcribe(pipeline)

    assert result["language"] == "pl"
    assert pipeline.tokenizer.language_code == "pl"


def test_the_task_is_kept_when_only_the_language_changes():
    pipeline = _pipeline(language="pl", task="translate")

    _transcribe(pipeline, language="en")

    assert pipeline.tokenizer.language_code == "en"
    assert pipeline.tokenizer.task == pipeline.tokenizer.translate


def test_another_task_with_the_same_language():
    pipeline = _pipeline(language="pl")

    _transcribe(pipeline, task="translate")

    assert pipeline.tokenizer.language_code == "pl"
    assert pipeline.tokenizer.task == pipeline.tokenizer.translate


def test_same_language_and_task_keep_the_tokenizer():
    pipeline = _pipeline(language="pl")
    tokenizer = pipeline.tokenizer

    _transcribe(pipeline, language="pl", task="transcribe")
    _transcribe(pipeline)

    assert pipeline.tokenizer is tokenizer


def test_english_only_model_keeps_its_tokenizer():
    pipeline = _pipeline(language="en", multilingual=False)
    tokenizer = pipeline.tokenizer

    _transcribe(pipeline)
    result = _transcribe(pipeline, task="transcribe")

    assert result["language"] == "en"
    assert pipeline.tokenizer is tokenizer
