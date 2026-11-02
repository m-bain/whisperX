"""Tests for per-stage device routing (cpu / cuda / mps / auto).

These mock the OS, the hardware checks and the model loaders, so they pass on any machine.
"""

import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

import whisperx.alignment as A
import whisperx.asr as asr
import whisperx.device as D
import whisperx.diarize as diarize


@contextmanager
def _machine(platform: str, cuda: bool, mps: bool, ct2_mps: bool = False):
    fake_ct2 = types.SimpleNamespace(get_mps_device_count=lambda: 1) if ct2_mps else types.SimpleNamespace()
    with mock.patch.object(sys, "platform", platform), mock.patch.multiple(
        "torch",
        cuda=mock.MagicMock(is_available=mock.MagicMock(return_value=cuda), device_count=mock.MagicMock(return_value=2 if cuda else 0)),
        backends=mock.MagicMock(mps=mock.MagicMock(is_available=mock.MagicMock(return_value=mps))),
    ), mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
        yield


MAC = dict(platform="darwin", cuda=False, mps=True)
MAC_NO_GPU = dict(platform="darwin", cuda=False, mps=False)
LINUX_CUDA = dict(platform="linux", cuda=True, mps=False)
LINUX_CPU = dict(platform="linux", cuda=False, mps=False)


@pytest.mark.parametrize(
    "machine,expected",
    [(MAC, "mps"), (MAC_NO_GPU, "cpu"), (LINUX_CUDA, "cuda"), (LINUX_CPU, "cpu"), (dict(platform="win32", cuda=True, mps=False), "cuda")],
)
def test_auto_order_is_per_os(machine, expected):
    with _machine(**machine):
        assert D.resolve_device("auto") == expected


def test_auto_never_picks_cuda_on_macos():
    with _machine("darwin", cuda=True, mps=False):
        assert D.resolve_device("auto") == "cpu"


def test_auto_never_picks_mps_off_macos():
    with _machine("linux", cuda=False, mps=True):
        assert D.resolve_device("auto") == "cpu"


@pytest.mark.parametrize(
    "machine,device",
    [(MAC, "cuda"), (MAC, "cuda:1"), (MAC_NO_GPU, "mps"), (LINUX_CUDA, "mps"), (LINUX_CPU, "cuda")],
)
def test_missing_device_is_an_error(machine, device):
    with _machine(**machine), pytest.raises(D.DeviceError, match="no .* available"):
        D.resolve_device(device)


@pytest.mark.parametrize("device", ["tpu", "cuda:x"])
def test_invalid_device_is_an_error(device):
    with _machine(**LINUX_CPU), pytest.raises(D.DeviceError, match="Invalid device"):
        D.resolve_device(device)


@pytest.mark.parametrize("device", ["xpu", "xpu:1"])
def test_other_torch_device_types_pass_through(device):
    with _machine(**LINUX_CPU):
        assert D.resolve_device(device) == device
        assert D.stage_device("VAD", device, "cpu", None, "cpu", "") == device


def test_cuda_index_must_exist():
    with _machine(**LINUX_CUDA):
        assert D.resolve_device("cuda:1") == "cuda:1"
        with pytest.raises(D.DeviceError, match="has 2 CUDA GPU"):
            D.resolve_device("cuda:2")


@pytest.mark.parametrize("machine,device", [(MAC, "cpu"), (MAC, "mps"), (LINUX_CUDA, "cuda"), (LINUX_CUDA, "cuda:1")])
def test_existing_devices_pass_through(machine, device):
    with _machine(**machine):
        assert D.resolve_device(device) == device


def test_stage_inherits_and_uses_supported_device():
    with _machine(**MAC):
        assert D.stage_device("VAD", None, "mps", ("cpu", "mps"), "cpu", "") == "mps"
        assert D.stage_device("VAD", None, "auto", ("cpu", "mps"), "cpu", "") == "mps"


def test_stage_own_device_overrides_inherited():
    with _machine(**MAC):
        assert D.stage_device("VAD", "cpu", "mps", ("cpu", "mps"), "cpu", "") == "cpu"
        assert D.stage_device("VAD", "mps", "cpu", ("cpu", "mps"), "cpu", "") == "mps"


def test_stage_falls_back_to_cpu_when_unsupported_device_is_inherited():
    with _machine(**MAC), mock.patch.object(D.logger, "warning") as warning:
        assert D.stage_device("ASR", None, "mps", ("cpu", "cuda"), "cpu", " (hint)") == "cpu"
        assert D.stage_device("ASR", None, "auto", ("cpu", "cuda"), "cpu", " (hint)") == "cpu"
    warning.assert_called_with("ASR cannot run on mps (hint); running it on cpu")
    assert warning.call_count == 2


def test_stage_inherited_fallback_can_be_a_gpu():
    with _machine(**MAC):
        assert D.stage_device("ASR", None, "cpu", ("mps",), "mps", "") == "mps"
    with _machine(**LINUX_CUDA), pytest.raises(D.DeviceError, match="Apple GPU"):
        D.stage_device("ASR", None, "cpu", ("mps",), "mps", "")


def test_stage_auto_set_on_the_stage_falls_back_too():
    with _machine(**MAC):
        assert D.stage_device("ASR", "auto", "cpu", ("cpu", "cuda"), "cpu", "") == "cpu"


def test_stage_explicit_unsupported_device_is_an_error_with_hint():
    with _machine(**MAC), pytest.raises(D.DeviceError, match=r"ASR cannot run on mps \(hint\)"):
        D.stage_device("ASR", "mps", "cpu", ("cpu", "cuda"), "cpu", " (hint)")


@pytest.mark.parametrize("own,inherited", [("cuda", None), (None, "cuda")])
def test_stage_missing_device_is_an_error_whether_explicit_or_inherited(own, inherited):
    with _machine(**MAC), pytest.raises(D.DeviceError, match="no CUDA GPU"):
        D.stage_device("VAD", own, inherited, ("cpu", "cuda", "mps"), "cpu", "")


def _fake_whisper_model():
    model = mock.MagicMock()
    model.model.is_multilingual = True
    return model


def _load_model(machine, **kwargs):
    with _machine(**machine), mock.patch.object(
        asr, "WhisperModel", return_value=_fake_whisper_model()
    ) as whisper_cls, mock.patch.object(asr, "Pyannote") as vad_cls, mock.patch.object(asr, "Silero") as silero_cls:
        asr.load_model("tiny", language="en", **kwargs)
    vad_device = vad_cls.call_args.args[0] if vad_cls.called else None
    return whisper_cls.call_args.kwargs, vad_device, silero_cls


def test_load_model_mps_runs_vad_on_mps_and_inherited_asr_on_cpu():
    asr_kwargs, vad_device, _ = _load_model(MAC, device="mps")
    assert asr_kwargs["device"] == "cpu"
    assert asr_kwargs["compute_type"] == "float32"
    assert vad_device == torch.device("mps")


def test_load_model_auto_on_mac_matches_mps():
    asr_kwargs, vad_device, _ = _load_model(MAC, device="auto")
    assert asr_kwargs["device"] == "cpu"
    assert vad_device == torch.device("mps")


def test_load_model_cpu_is_unchanged():
    asr_kwargs, vad_device, _ = _load_model(MAC, device="cpu")
    assert asr_kwargs["device"] == "cpu"
    assert vad_device == torch.device("cpu")


def test_load_model_cuda_uses_device_index_for_vad():
    asr_kwargs, vad_device, _ = _load_model(LINUX_CUDA, device="cuda", device_index=1)
    assert asr_kwargs["device"] == "cuda"
    assert asr_kwargs["compute_type"] == "float16"
    assert vad_device == torch.device("cuda:1")


def test_load_model_indexed_asr_device_is_split_for_ctranslate2():
    asr_kwargs, vad_device, _ = _load_model(LINUX_CUDA, device="cuda", asr_device="cuda:1")
    assert (asr_kwargs["device"], asr_kwargs["device_index"]) == ("cuda", 1)
    assert vad_device == torch.device("cuda:0")


def test_load_model_stage_devices_override_device():
    asr_kwargs, vad_device, _ = _load_model(MAC, device="cpu", vad_device="mps", asr_device="cpu")
    assert asr_kwargs["device"] == "cpu"
    assert vad_device == torch.device("mps")


def test_load_model_explicit_asr_mps_with_faster_whisper_is_an_error():
    with pytest.raises(D.DeviceError, match="ASR cannot run on mps.*CTranslate2"):
        _load_model(MAC, device="cpu", asr_device="mps")


def test_load_model_uses_ctranslate2_metal_build_when_present():
    with _machine(**MAC, ct2_mps=True), mock.patch.object(
        asr, "WhisperModel", return_value=_fake_whisper_model()
    ) as whisper_cls, mock.patch.object(asr, "Pyannote"):
        asr.load_model("tiny", device="mps", asr_device="mps", language="en")
    assert whisper_cls.call_args.kwargs["device"] == "mps"


def test_load_model_missing_device_fails_before_loading_anything():
    with _machine(**MAC), mock.patch.object(asr, "WhisperModel") as whisper_cls, mock.patch.object(
        asr, "Pyannote"
    ) as vad_cls, pytest.raises(D.DeviceError, match="no CUDA GPU"):
        asr.load_model("tiny", device="cuda", language="en")
    whisper_cls.assert_not_called()
    vad_cls.assert_not_called()


def test_load_model_silero_vad_inherits_mps_as_cpu():
    asr_kwargs, _, silero_cls = _load_model(MAC, device="mps", vad_method="silero")
    assert silero_cls.called
    assert asr_kwargs["device"] == "cpu"


def test_load_model_silero_vad_explicit_mps_is_an_error():
    with pytest.raises(D.DeviceError, match="VAD cannot run on mps.*Silero"):
        _load_model(MAC, device="cpu", vad_method="silero", vad_device="mps")


def test_load_model_ignores_vad_device_for_a_manually_assigned_vad_model():
    with _machine(**MAC), mock.patch.object(asr, "WhisperModel", return_value=_fake_whisper_model()):
        pipeline = asr.load_model("tiny", device="cpu", language="en", vad_model=mock.MagicMock())
    assert isinstance(pipeline, asr.FasterWhisperPipeline)


def _load_align_model(machine, device):
    fake_processor = mock.MagicMock()
    fake_processor.tokenizer.get_vocab.return_value = {"<pad>": 0, "a": 1}
    fake_model = mock.MagicMock()
    with _machine(**machine), mock.patch.object(A, "torchaudio", None), mock.patch.object(
        A, "Wav2Vec2Processor"
    ) as proc_cls, mock.patch.object(A, "Wav2Vec2ForCTC") as model_cls:
        proc_cls.from_pretrained.return_value = fake_processor
        model_cls.from_pretrained.return_value = fake_model
        A.load_align_model(language_code="en", device=device)
    return fake_model


def test_load_align_model_resolves_auto_device():
    _load_align_model(MAC, "auto").to.assert_called_once_with("mps")


def test_load_align_model_missing_device_is_an_error():
    with pytest.raises(D.DeviceError, match="no CUDA GPU"):
        _load_align_model(MAC, "cuda")


def test_diarization_pipeline_resolves_auto_device():
    with _machine(**MAC), mock.patch.object(diarize, "Pipeline") as pipeline_cls:
        diarize.DiarizationPipeline(device="auto")
    pipeline_cls.from_pretrained.return_value.to.assert_called_once_with(torch.device("mps"))


def test_diarization_pipeline_missing_device_is_an_error():
    with _machine(**LINUX_CPU), mock.patch.object(diarize, "Pipeline"), pytest.raises(D.DeviceError, match="Apple GPU"):
        diarize.DiarizationPipeline(device="mps")
