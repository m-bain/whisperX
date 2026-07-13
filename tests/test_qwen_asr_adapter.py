from types import SimpleNamespace

import numpy as np

from whisperx.asr_qwen import (
    QwenAsrPipeline,
    qwen_language_to_whisper_code,
    whisper_language_code_to_qwen,
)


class DummyVad:
    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size, onset, offset):
        del chunk_size, onset, offset
        return segments

    def __call__(self, payload):
        del payload
        return [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ]


class DummyQwenModel:
    def transcribe(self, audio, language=None, return_time_stamps=False):
        del audio, language, return_time_stamps
        return [
            SimpleNamespace(text="hello", language="English"),
            SimpleNamespace(text="wereld", language="Dutch"),
        ]


def test_language_mapping_helpers_cover_key_cases():
    assert whisper_language_code_to_qwen("en") == "English"
    assert whisper_language_code_to_qwen("yue") == "Cantonese"
    assert whisper_language_code_to_qwen("tl") == "Filipino"

    assert qwen_language_to_whisper_code("English") == "en"
    assert qwen_language_to_whisper_code("Cantonese") == "yue"
    assert qwen_language_to_whisper_code("Chinese,English") == "zh"


def test_qwen_pipeline_transcribe_returns_whisperx_schema():
    pipeline = QwenAsrPipeline(
        model=DummyQwenModel(),
        vad=DummyVad(),
        vad_params={"vad_onset": 0.5, "vad_offset": 0.3},
        language=None,
    )
    audio = np.zeros(2 * 16000, dtype=np.float32)

    out = pipeline.transcribe(audio, batch_size=2, chunk_size=30, verbose=False)

    assert out["language"] == "en"
    assert len(out["segments"]) == 2
    assert out["segments"][0]["text"] == "hello"
    assert out["segments"][0]["start"] == 0.0
    assert out["segments"][0]["end"] == 1.0
    assert out["segments"][1]["text"] == "wereld"
