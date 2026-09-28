from types import SimpleNamespace

import pytest
import torch

from whisperx import alignment


ALIGN_METADATA = {
    "dictionary": {"<pad>": 0, "a": 1, "b": 2},
    "language": "en",
    "type": "torchaudio",
}


class FixedEmissionModel(torch.nn.Module):
    def forward(self, waveform, lengths=None):
        logits = torch.tensor(
            [[
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 5.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
                [5.0, 0.0, 0.0],
            ]],
            device=waveform.device,
        ).repeat(waveform.shape[0], 1, 1)
        output_lengths = torch.full(
            (waveform.shape[0],), logits.shape[1], device=waveform.device
        )
        return logits, output_lengths


def test_batch_size_one_matches_serial(monkeypatch):
    logits, _ = FixedEmissionModel()(torch.zeros(1, 960))
    emission = torch.log_softmax(logits[0], dim=-1)
    monkeypatch.setattr(
        alignment, "_compute_emission", lambda *args, **kwargs: emission
    )
    monkeypatch.setattr(
        alignment, "_compute_emission_batch", lambda waveforms, *args: [emission] * len(waveforms)
    )
    transcript = [{"start": 0.0, "end": 0.06, "text": "ab"}]

    serial = alignment.align(
        transcript, None, ALIGN_METADATA, torch.zeros(960), "cpu",
        return_char_alignments=True,
    )
    batched = alignment.align_batch(
        [transcript], None, ALIGN_METADATA, [torch.zeros(960)], "cpu", 1,
        return_char_alignments=True,
    )[0]

    assert batched == serial


def test_align_batch_sorts_by_duration_and_restores_input_order(monkeypatch):
    acoustic_batches = []

    def compute_batch(waveforms, model, model_type, device):
        acoustic_batches.append([waveform.shape[-1] for waveform in waveforms])
        return [torch.empty(1, 3) for _ in waveforms]

    def finish(segment, *args, **kwargs):
        return [{"start": segment["start"], "end": segment["end"],
                 "text": segment["text"], "words": []}]

    monkeypatch.setattr(alignment, "_compute_emission_batch", compute_batch)
    monkeypatch.setattr(alignment, "_finish_alignment_segment", finish)
    transcripts = [
        [{"start": 0.0, "end": 0.03, "text": "a"}],
        [{"start": 0.0, "end": 0.01, "text": "a"}],
        [{"start": 0.0, "end": 0.02, "text": "b"}],
    ]

    results = alignment.align_batch(
        transcripts,
        model=None,
        align_model_metadata=ALIGN_METADATA,
        audio=[torch.zeros(480), torch.zeros(160), torch.zeros(320)],
        device="cpu",
        batch_size=2,
    )

    assert acoustic_batches == [[160, 320], [480]]
    assert [result["segments"][0]["text"] for result in results] == ["a", "a", "b"]


def test_align_batch_validates_inputs():
    with pytest.raises(ValueError, match="same number"):
        alignment.align_batch([], None, ALIGN_METADATA, [torch.zeros(1)], "cpu", 1)
    with pytest.raises(ValueError, match="at least 1"):
        alignment.align_batch([], None, ALIGN_METADATA, [], "cpu", 0)


@pytest.mark.parametrize("feat_extract_norm", ["group", None, "custom"])
def test_huggingface_batch_falls_back_for_unvalidated_norms(
    monkeypatch, feat_extract_norm
):
    model = SimpleNamespace(config=SimpleNamespace(feat_extract_norm=feat_extract_norm))
    calls = []

    def compute_serial(waveform, model, model_type, device):
        calls.append(waveform.shape)
        return torch.zeros(waveform.shape[-1], 2)

    monkeypatch.setattr(alignment, "_compute_emission", compute_serial)
    emissions = alignment._compute_emission_batch(
        [torch.zeros(1, 399), torch.zeros(800)], model, "huggingface", "cpu"
    )

    assert calls == [torch.Size([1, 399]), torch.Size([1, 800])]
    assert [emission.shape for emission in emissions] == [(399, 2), (800, 2)]
