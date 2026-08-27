"""Tests for torchaudio-optional alignment model loading."""

import whisperx.alignment as A


def test_default_torch_models_have_hf_fallback():
    """每个 torchaudio.pipelines 默认对齐模型都应有 HF 等价兜底。

    该兜底让没有 torchaudio（如昇腾/ARM aarch64，无 torchaudio wheel）的
    平台也能用对应的 Hugging Face wav2vec2 checkpoint 执行 forced alignment。
    """
    assert set(A.DEFAULT_ALIGN_MODELS_TORCH.values()) == set(
        A.TORCHAUDIO_PIPELINE_TO_HF.keys()
    )


def test_hf_fallback_names_are_valid_checkpoints():
    """HF 兜底名应为 facebook 官方 wav2vec2 checkpoint。"""
    for name in A.TORCHAUDIO_PIPELINE_TO_HF.values():
        assert name.startswith("facebook/"), name