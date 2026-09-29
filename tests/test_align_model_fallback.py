"""Tests for torchaudio-optional alignment model loading."""

from unittest import mock

import whisperx.alignment as A


def test_default_torch_models_have_hf_fallback():
    """Every default torchaudio.pipelines align model should have an HF
    equivalent, so platforms without torchaudio wheels (e.g. Ascend /
    Linux aarch64) can still run forced alignment via Hugging Face."""
    assert set(A.DEFAULT_ALIGN_MODELS_TORCH.values()).issubset(
        A.TORCHAUDIO_PIPELINE_TO_HF
    )


def test_hf_fallback_names_are_valid_checkpoints():
    """HF fallback names should be official facebook wav2vec2 checkpoints."""
    for name in A.TORCHAUDIO_PIPELINE_TO_HF.values():
        assert name.startswith("facebook/"), name


def test_load_align_model_falls_back_to_hf_when_torchaudio_missing():
    """With torchaudio absent, a torchaudio.pipelines default model name must
    be transparently remapped to its HF checkpoint and loaded via
    Wav2Vec2 (the HuggingFace path), not crash."""
    fake_vocab = {"<pad>": 0, "a": 1, "b": 2}
    fake_processor = mock.MagicMock()
    fake_processor.tokenizer.get_vocab.return_value = fake_vocab
    fake_model = mock.MagicMock()

    with mock.patch.object(A, "torchaudio", None), mock.patch.object(
        A, "Wav2Vec2Processor"
    ) as proc_cls, mock.patch.object(A, "Wav2Vec2ForCTC") as model_cls:
        proc_cls.from_pretrained.return_value = fake_processor
        model_cls.from_pretrained.return_value = fake_model

        model, metadata = A.load_align_model(
            language_code="en", device="cpu"
        )

    assert model is fake_model.to.return_value  # returned after .to(device)
    assert metadata["type"] == "huggingface"
    assert metadata["dictionary"] == {c.lower(): i for c, i in fake_vocab.items()}
    # the torchaudio bundle name was remapped to the HF checkpoint
    remapped_name = proc_cls.from_pretrained.call_args.args[0]
    assert remapped_name == A.TORCHAUDIO_PIPELINE_TO_HF["WAV2VEC2_ASR_BASE_960H"]
    assert model_cls.from_pretrained.call_args.args[0] == remapped_name
