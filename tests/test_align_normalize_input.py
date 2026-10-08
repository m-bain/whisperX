"""align(normalize_input=True) feeds Hugging Face models the input their feature extractor would give them."""

import sys
import warnings

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2FeatureExtractor

from whisperx import alignment
from whisperx.alignment import align, normalize_waveform

DICTIONARY = {"<pad>": 0, "a": 1, "c": 2, "t": 3, "|": 4}


class _Recorder:
    """An alignment model that records its input and returns a flat emission."""

    def __init__(self, model_type):
        self.model_type = model_type
        self.inputs = []

    def __call__(self, waveform, lengths=None):
        self.inputs.append(waveform.clone())
        emission = torch.zeros(1, max(waveform.shape[-1] // 320, 1), len(DICTIONARY))
        if self.model_type == "torchaudio":
            return emission, None
        return type("Output", (), {"logits": emission})()


def _align(audio, model_type="huggingface", do_normalize=True, start=0.0, end=None, **kwargs):
    model = _Recorder(model_type)
    metadata = {"language": "en", "dictionary": DICTIONARY, "type": model_type}
    if do_normalize is not None:
        metadata["do_normalize"] = do_normalize
    end = audio.shape[-1] / 16000 if end is None else end
    align([{"text": "a cat", "start": start, "end": end}], model, metadata, audio, "cpu", **kwargs)
    return model.inputs[0][0] if model.inputs else None


def _quiet_audio(seconds=2.0):
    torch.manual_seed(0)
    return 0.003 * torch.randn(int(seconds * 16000)) + 0.001


@pytest.mark.parametrize("samples", [400, 16000, 48123])
def test_normalize_waveform_matches_the_feature_extractor(samples):
    rng = np.random.default_rng(samples)
    x = (0.01 * rng.standard_normal(samples) + 0.002).astype(np.float32)
    extractor = Wav2Vec2FeatureExtractor(do_normalize=True)

    expected = extractor(x, sampling_rate=16000, return_tensors="np").input_values[0]
    actual = normalize_waveform(torch.from_numpy(x).unsqueeze(0))[0].numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_raw_input_by_default():
    audio = _quiet_audio()

    seen = _align(audio)

    assert torch.equal(seen, audio)


def test_normalized_input_when_asked():
    audio = _quiet_audio()

    seen = _align(audio, normalize_input=True)

    assert torch.allclose(seen, normalize_waveform(audio.unsqueeze(0))[0])
    assert seen.std().item() > 0.9  # the raw input has std 0.003


@pytest.mark.parametrize("model_type, do_normalize", [
    ("huggingface", False),  # the model's feature extractor does not normalize
    ("huggingface", None),   # metadata built by hand, without the key
    ("torchaudio", True),    # torchaudio bundles handle their own input
])
def test_other_models_keep_the_raw_input(model_type, do_normalize):
    audio = _quiet_audio()

    seen = _align(audio, model_type, do_normalize, normalize_input=True)

    assert torch.equal(seen, audio)


def test_each_segment_is_normalized_on_its_own():
    torch.manual_seed(0)
    audio = torch.cat([0.3 * torch.randn(16000), 0.003 * torch.randn(16000)])

    seen = _align(audio, start=1.0, end=2.0, normalize_input=True)

    assert torch.allclose(seen, normalize_waveform(audio[16000:].unsqueeze(0))[0])


def test_short_segment_is_normalized_before_padding():
    audio = _quiet_audio(seconds=0.015)  # 240 samples, padded to 400

    seen = _align(audio, normalize_input=True)

    assert seen.shape[-1] == 400
    assert torch.allclose(seen[:240], normalize_waveform(audio.unsqueeze(0))[0])
    assert torch.all(seen[240:] == 0)


def test_empty_segment_does_not_warn():
    audio = _quiet_audio()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _align(audio, start=0.5, end=0.5, normalize_input=True)


@pytest.mark.parametrize("do_normalize", [True, False])
def test_load_align_model_reads_do_normalize(monkeypatch, do_normalize):
    vocab = {"<pad>": 0, "a": 1}
    processor = type("Processor", (), {
        "feature_extractor": Wav2Vec2FeatureExtractor(do_normalize=do_normalize),
        "tokenizer": type("Tokenizer", (), {"get_vocab": staticmethod(lambda: vocab)})(),
    })()
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(alignment.Wav2Vec2Processor, "from_pretrained", staticmethod(lambda *a, **k: processor))
    monkeypatch.setattr(alignment.Wav2Vec2ForCTC, "from_pretrained", staticmethod(lambda *a, **k: model))

    _, metadata = alignment.load_align_model("pl", "cpu")

    assert metadata["type"] == "huggingface"
    assert metadata["do_normalize"] is do_normalize


@pytest.mark.parametrize("argv, expected", [([], False), (["--align_normalize_input"], True)])
def test_cli_flag(monkeypatch, argv, expected):
    import whisperx.transcribe
    from whisperx.__main__ import cli

    seen = {}
    monkeypatch.setattr(whisperx.transcribe, "transcribe_task", lambda args, parser: seen.update(args))
    monkeypatch.setattr(sys, "argv", ["whisperx", "audio.wav", *argv])

    cli()

    assert seen["align_normalize_input"] is expected


def test_cli_flag_reaches_align(monkeypatch, tmp_path):
    import whisperx.transcribe as wt
    from whisperx.__main__ import cli

    seen = {}
    segments = [{"text": "a", "start": 0.0, "end": 1.0}]
    model = type("Model", (), {"transcribe": lambda self, audio, **kw: {"segments": segments, "language": "en"}})()
    monkeypatch.setattr(wt, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(wt, "load_audio", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(wt, "load_align_model", lambda *a, **k: (object(), {"language": "en"}))
    monkeypatch.setattr(wt, "align", lambda *a, **k: seen.update(k) or {"segments": [], "word_segments": []})
    monkeypatch.setattr(wt, "get_writer", lambda *a, **k: lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["whisperx", "audio.wav", "--output_dir", str(tmp_path), "--align_normalize_input"])

    cli()

    assert seen["normalize_input"] is True
