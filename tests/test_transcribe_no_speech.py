"""transcribe() when VAD finds no speech in the audio."""

import numpy as np
import pytest
from faster_whisper.tokenizer import Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel

from whisperx.asr import FasterWhisperPipeline
from whisperx.vads import Vad

SPECIAL_TOKENS = ["<|transcribe|>", "<|translate|>", "<|en|>"]


class _Vad(Vad):
    def __init__(self, chunks):
        super().__init__(0.5)
        self.chunks = chunks

    @staticmethod
    def preprocess_audio(audio):
        return audio

    def merge_chunks(self, segments, chunk_size, onset, offset):
        return self.chunks

    def __call__(self, inputs):
        return []


class _Pipeline(FasterWhisperPipeline):
    """Stands in for the model. Like transformers 5.0 to 5.13, whose
    Pipeline.__call__ turned a generator input into a list and read its first
    item, it raises IndexError when called with an empty input."""

    def __call__(self, inputs, batch_size=None, num_workers=None):
        inputs = list(inputs)
        inputs[0]
        self.calls.append(len(inputs))
        return [{"text": f"chunk {i}", "avg_logprob": -0.5} for i in range(len(inputs))]


def _pipeline(chunks):
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS + ["[UNK]"])}
    hf_tokenizer = HFTokenizer(WordLevel(vocab, unk_token="[UNK]"))
    pipeline = _Pipeline(
        model=None,
        vad=_Vad(chunks),
        vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
        options=None,
        tokenizer=Tokenizer(hf_tokenizer, True, task="transcribe", language="en"),
        language="en",
    )
    pipeline.calls = []
    return pipeline


@pytest.mark.parametrize(
    "batch_size, interleaved_context", [(1, False), (2, False), (2, True)]
)
def test_no_speech_returns_no_segments(batch_size, interleaved_context):
    pipeline = _pipeline(chunks=[])
    result = pipeline.transcribe(
        np.zeros(16000, dtype=np.float32),
        batch_size=batch_size,
        interleaved_context=interleaved_context,
    )
    assert result == {"segments": [], "language": "en"}
    assert pipeline.calls == []


def test_speech_chunks_still_go_through_the_pipeline():
    chunks = [{"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.5}]
    pipeline = _pipeline(chunks=chunks)
    result = pipeline.transcribe(np.zeros(3 * 16000, dtype=np.float32), batch_size=2)
    assert pipeline.calls == [2]
    assert result == {
        "segments": [
            {"text": "chunk 0", "start": 0.0, "end": 1.0, "avg_logprob": -0.5},
            {"text": "chunk 1", "start": 1.5, "end": 2.5, "avg_logprob": -0.5},
        ],
        "language": "en",
    }
