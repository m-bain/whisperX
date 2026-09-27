"""Exercise ASR batching without loading a speech model."""

import logging
import multiprocessing

import pytest
import torch

from whisperx.asr import FasterWhisperPipeline


class EchoPipeline(FasterWhisperPipeline):
    def __init__(self):
        self.framework = "pt"
        self.device = torch.device("cpu")

    def preprocess(self, audio):
        return audio

    def _forward(self, model_inputs):
        return {"index": model_inputs["inputs"]}


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="The existing ASR collate function requires fork for worker processes",
)
@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_asr_iterator_emits_each_segment_once(monkeypatch, caplog, num_workers, batch_size):
    # Select Linux's worker context explicitly on platforms that default to spawn.
    dataloader_class = torch.utils.data.DataLoader

    def dataloader(*args, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            kwargs["multiprocessing_context"] = "fork"
        return dataloader_class(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", dataloader)
    monkeypatch.setattr(logging.getLogger("whisperx"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="whisperx.asr")

    inputs = ({"inputs": torch.tensor([index])} for index in range(5))
    iterator = EchoPipeline().get_iterator(
        inputs, num_workers, batch_size, {}, {}, {}
    )

    assert [output["index"].item() for output in iterator] == list(range(5))
    assert ("num_workers=1" in caplog.text) == (num_workers > 1)
