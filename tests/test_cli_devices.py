"""Tests that the CLI routes --device and the per-stage device flags to the right stages."""

import sys
from unittest import mock

import numpy as np
import pytest

import whisperx.transcribe as T
from whisperx.__main__ import cli


def _run_cli(tmp_path, *flags, platform="darwin", cuda=False, mps=True):
    model = mock.MagicMock()
    model.transcribe.return_value = {"segments": [{"text": " hi", "start": 0.0, "end": 1.0}], "language": "en"}
    stages = mock.MagicMock()
    stages.load_align_model.return_value = (mock.MagicMock(), {"language": "en"})
    stages.align.return_value = {"segments": [], "word_segments": []}
    with mock.patch.object(sys, "platform", platform), mock.patch(
        "torch.backends.mps.is_available", return_value=mps
    ), mock.patch("torch.cuda.is_available", return_value=cuda), mock.patch.object(
        T, "load_model", return_value=model
    ) as load_model, mock.patch.object(
        T, "load_audio", return_value=np.zeros(16000, dtype=np.float32)
    ), mock.patch.object(T, "load_align_model", stages.load_align_model), mock.patch.object(
        T, "align", stages.align
    ), mock.patch.object(T, "DiarizationPipeline", stages.DiarizationPipeline), mock.patch.object(
        T, "assign_word_speakers", side_effect=lambda df, result, emb: result
    ), mock.patch.object(T, "get_writer", return_value=lambda *a: None), mock.patch.object(
        sys, "argv", ["whisperx", "audio.wav", "--diarize", "--language", "en", "-o", str(tmp_path), *flags]
    ):
        cli()
    load_kwargs = load_model.call_args.kwargs
    return (
        load_kwargs["device"],
        load_kwargs["vad_device"],
        load_kwargs["asr_device"],
        stages.load_align_model.call_args.args[1],
        stages.align.call_args.args[4],
        stages.DiarizationPipeline.call_args.kwargs["device"],
    )


@pytest.mark.parametrize(
    "flags,expected",
    [
        # (device, vad_device, asr_device) go to load_model unresolved; align / diarize are resolved
        ([], ("auto", None, None, "mps", "mps", "mps")),
        (["--device", "auto"], ("auto", None, None, "mps", "mps", "mps")),
        (["--device", "mps"], ("mps", None, None, "mps", "mps", "mps")),
        (["--device", "cpu"], ("cpu", None, None, "cpu", "cpu", "cpu")),
        (["--device", "cpu", "--diarize_device", "mps"], ("cpu", None, None, "cpu", "cpu", "mps")),
        (["--device", "cpu", "--align_device", "mps", "--diarize_device", "auto"], ("cpu", None, None, "mps", "mps", "mps")),
        (["--vad_device", "mps", "--asr_device", "cpu"], ("auto", "mps", "cpu", "mps", "mps", "mps")),
    ],
)
def test_cli_routes_devices_per_stage(tmp_path, flags, expected):
    assert _run_cli(tmp_path, *flags) == expected


def test_cli_passes_indexed_and_other_torch_devices_through(tmp_path):
    with mock.patch("torch.cuda.device_count", return_value=2):
        result = _run_cli(tmp_path, "--diarize_device", "cuda:1", "--align_device", "xpu", platform="linux", cuda=True, mps=False)
    assert result[3:] == ("xpu", "xpu", "cuda:1")


def test_cli_omitted_device_on_linux_with_cuda_is_cuda(tmp_path):
    assert _run_cli(tmp_path, platform="linux", cuda=True, mps=False)[3:] == ("cuda", "cuda", "cuda")


@pytest.mark.parametrize(
    "flags",
    [["--device", "cuda"], ["--align_device", "cuda"], ["--diarize_device", "cuda"]],
)
def test_cli_missing_device_exits_with_a_clear_error(tmp_path, flags, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_cli(tmp_path, *flags)
    assert exc.value.code == 2
    assert "no CUDA GPU is available" in capsys.readouterr().err


def test_cli_explicit_stage_device_the_stage_cannot_use_exits_with_hint(tmp_path, capsys):
    # Unlike _run_cli, this keeps the real load_model, whose stage check fails before any model loads
    with pytest.raises(SystemExit) as exc, mock.patch.object(sys, "platform", "darwin"), mock.patch(
        "torch.backends.mps.is_available", return_value=True
    ), mock.patch("torch.cuda.is_available", return_value=False), mock.patch.object(
        T, "load_audio", return_value=np.zeros(16000, dtype=np.float32)
    ), mock.patch.object(
        sys, "argv", ["whisperx", "audio.wav", "-o", str(tmp_path), "--asr_device", "mps"]
    ), mock.patch.dict(sys.modules, {"ctranslate2": type(sys)("ctranslate2")}):
        cli()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "ASR cannot run on mps" in err
    assert "CTranslate2" in err
