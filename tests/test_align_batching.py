import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from whisperx.alignment import _batched_emissions


def tiny_model(feat_extract_norm):
    torch.manual_seed(0)
    config = Wav2Vec2Config(
        vocab_size=12, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        feat_extract_norm=feat_extract_norm, do_stable_layer_norm=True,
        conv_dim=(32, 32, 32), conv_kernel=(10, 3, 3), conv_stride=(5, 2, 2),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
    )
    return Wav2Vec2ForCTC(config).eval()


def test_batched_emissions_match_one_segment_at_a_time():
    model = tiny_model("layer")
    audio = torch.randn(1, 16000 * 6)
    spans = [(0, 0, 16000), (1, 8000, 52000), (2, 20000, 20100), (3, 30000, 60000), (4, 1000, 9000)]
    batched = _batched_emissions(model, "huggingface", audio, spans, "cpu", batch_size=3)
    assert set(batched) == {0, 1, 2, 3, 4}
    for sdx, f1, f2 in spans:
        segment = torch.nn.functional.pad(audio[:, f1:f2], (0, max(0, 400 - (f2 - f1))))
        with torch.inference_mode():
            alone = torch.log_softmax(model(segment).logits, dim=-1)[0]
        assert batched[sdx].shape == alone.shape
        torch.testing.assert_close(batched[sdx], alone, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model_type,norm,batch_size", [("torchaudio", "layer", 8), ("huggingface", "group", 8), ("huggingface", "layer", 1)])
def test_batching_not_used_when_it_would_change_results(model_type, norm, batch_size):
    model = tiny_model(norm)
    assert _batched_emissions(model, model_type, torch.randn(1, 16000), [(0, 0, 8000)], "cpu", batch_size) == {}


def test_batches_are_split_when_the_gpu_runs_out_of_memory():
    model = tiny_model("layer")
    audio = torch.randn(1, 16000 * 4)
    spans = [(i, i * 8000, i * 8000 + 20000) for i in range(6)]
    real_forward = model.forward
    calls = []

    def forward(input_values, **kwargs):
        calls.append(input_values.shape[0])
        if input_values.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return real_forward(input_values, **kwargs)

    model.forward = forward
    batched = _batched_emissions(model, "huggingface", audio, spans, "cpu", batch_size=8)
    assert set(batched) == set(range(6))
    assert max(calls) == 6 and 1 <= calls[-1] <= 2  # tried 6, then halves


def test_batch_audio_length_is_capped(monkeypatch):
    import whisperx.alignment as A
    monkeypatch.setattr(A, "ALIGN_BATCH_MAX_SECONDS", 2)
    model = tiny_model("layer")
    sizes = []
    real_forward = model.forward
    model.forward = lambda x, **k: (sizes.append(x.shape[0]), real_forward(x, **k))[1]
    spans = [(i, 0, 16000) for i in range(5)]  # five 1-second segments, at most two seconds per batch
    A._batched_emissions(model, "huggingface", torch.randn(1, 16000), spans, "cpu", batch_size=8)
    assert max(sizes) == 2
