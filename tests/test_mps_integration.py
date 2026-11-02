"""Apple GPU integration tests: run real models on MPS and compare with CPU.

Skipped unless the machine has an Apple GPU.
These download models on first run. The speech clip is synthesized with macOS `say`:
the VAD model is numerically ill-conditioned on pure tones (tiny input changes move its
scores by >0.1 on CPU alone), so CPU/MPS parity is only meaningful on real speech.
"""

import shutil
import subprocess

import numpy as np
import pytest
import torch

from whisperx.audio import SAMPLE_RATE, load_audio

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")

SENTENCE = "The quick brown fox jumps over the lazy dog. Speech recognition should give the same words on every device."


@pytest.fixture(scope="module")
def speech(tmp_path_factory) -> np.ndarray:
    if shutil.which("say") is None:
        pytest.skip("needs macOS `say` to synthesize speech")
    path = tmp_path_factory.mktemp("speech") / "speech.aiff"
    subprocess.run(["say", "-o", str(path), SENTENCE], check=True)
    return load_audio(str(path))


def test_pyannote_vad_scores_match_cpu(speech):
    from whisperx.vads.pyannote import Pyannote

    scores = {}
    for device in ["cpu", "mps"]:
        vad = Pyannote(torch.device(device), vad_onset=0.5, vad_offset=0.363, chunk_size=30)
        scores[device] = vad({"waveform": Pyannote.preprocess_audio(speech), "sample_rate": SAMPLE_RATE}).data
    np.testing.assert_allclose(scores["mps"], scores["cpu"], atol=1e-3)


def test_alignment_emissions_match_cpu(speech):
    from whisperx.alignment import load_align_model

    audio = torch.from_numpy(speech)[None]
    emissions = {}
    for device in ["cpu", "mps"]:
        model, metadata = load_align_model("en", device)
        with torch.inference_mode():
            if metadata["type"] == "torchaudio":
                out, _ = model(audio.to(device))
            else:
                out = model(audio.to(device)).logits
        emissions[device] = torch.log_softmax(out, dim=-1).cpu().numpy()
    np.testing.assert_array_equal(emissions["mps"].argmax(-1), emissions["cpu"].argmax(-1))
    np.testing.assert_allclose(emissions["mps"], emissions["cpu"], atol=1e-2)
