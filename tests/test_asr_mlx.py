"""Tests for transcribing on mps with MLX; mlx and mlx_whisper are faked, so these run anywhere."""

import importlib.util
import shutil
import subprocess
import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch

def _fake_mlx_modules(transcribe):
    fake_model = mock.MagicMock(is_multilingual=True, num_languages=99)
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.float16 = "float16"
    mlx_core.default_device = lambda: "gpu"
    mlx = types.ModuleType("mlx")
    mlx.core = mlx_core
    mlx_whisper = types.ModuleType("mlx_whisper")
    mlx_whisper.transcribe = transcribe
    mlx_transcribe = types.ModuleType("mlx_whisper.transcribe")
    mlx_transcribe.ModelHolder = mock.MagicMock(get_model=mock.MagicMock(return_value=fake_model))
    return {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx_whisper": mlx_whisper,
        "mlx_whisper.transcribe": mlx_transcribe,
    }


def test_mlx_model_names_map_to_mlx_community_checkpoints():
    from whisperx.asr_mlx import MLX_MODELS

    assert MLX_MODELS["large-v3-turbo"] == "mlx-community/whisper-large-v3-turbo"
    assert MLX_MODELS["turbo"] == MLX_MODELS["large-v3-turbo"]
    assert all(repo.startswith("mlx-community/") for repo in MLX_MODELS.values())


def test_mlx_decode_options_pass_every_supported_option_through():
    from whisperx.asr_mlx import mlx_decode_options

    options = mlx_decode_options(
        {
            "temperatures": [0.0, 0.2],
            "best_of": 3,
            "length_penalty": 0.5,
            "suppress_tokens": [-1, 7],
            "initial_prompt": "Hi.",
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.0,
            "log_prob_threshold": -2.0,
            "no_speech_threshold": 0.5,
            "max_new_tokens": 100,
        },
        "default",
    )
    assert options == {
        "fp16": True,
        "temperature": (0.0, 0.2),
        "best_of": 3,
        "length_penalty": 0.5,
        "suppress_tokens": [-1, 7],
        "initial_prompt": "Hi.",
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.0,
        "logprob_threshold": -2.0,
        "no_speech_threshold": 0.5,
        "sample_len": 100,
    }


@pytest.mark.parametrize("compute_type,fp16", [("default", True), ("float16", True), ("int8", True), ("float32", False)])
def test_mlx_compute_type_maps_to_fp16(compute_type, fp16):
    from whisperx.asr_mlx import mlx_decode_options

    assert mlx_decode_options(None, compute_type)["fp16"] is fp16


def test_mlx_pipeline_returns_one_segment_per_vad_chunk():
    calls = []

    def fake_transcribe(audio, **kwargs):
        calls.append((len(audio), kwargs))
        return {"text": f" chunk {len(calls)}", "segments": [{"avg_logprob": -0.25}], "language": "en"}

    vad_chunks = [{"start": 0.0, "end": 1.5, "segments": []}, {"start": 2.0, "end": 3.0, "segments": []}]
    with mock.patch.dict(sys.modules, _fake_mlx_modules(fake_transcribe)), mock.patch(
        "whisperx.asr_mlx.mlx_model_path", return_value="/models/tiny"
    ), mock.patch("whisperx.asr_mlx.get_vad_segments", return_value=vad_chunks):
        from whisperx.asr_mlx import load_mlx_model

        pipeline = load_mlx_model(
            "tiny",
            vad_model=mock.MagicMock(),
            vad_method="pyannote",
            vad_options=None,
            vad_device="mps",
            device_index=None,
            asr_options=None,
            language="en",
            task="transcribe",
            compute_type="default",
            threads=None,
            download_root=None,
            local_files_only=False,
        )
        result = pipeline.transcribe(np.zeros(16000 * 4, dtype=np.float32), batch_size=8)

    assert result["language"] == "en"
    assert result["segments"] == [
        {"text": " chunk 1", "start": 0.0, "end": 1.5, "avg_logprob": -0.25},
        {"text": " chunk 2", "start": 2.0, "end": 3.0, "avg_logprob": -0.25},
    ]
    assert [n for n, _ in calls] == [24000, 16000]
    assert calls[0][1]["path_or_hf_repo"] == "/models/tiny"
    assert calls[0][1]["without_timestamps"] is True
    assert calls[0][1]["condition_on_previous_text"] is False
    assert calls[0][1]["language"] == "en" and calls[0][1]["task"] == "transcribe"


def _load_model_on(device, modules=None, **kwargs):
    import whisperx.asr as asr

    with mock.patch.object(asr, "Pyannote") as vad_cls, mock.patch.object(asr, "WhisperModel") as whisper_cls, mock.patch(
        "whisperx.asr_mlx.mlx_model_path", return_value="/models/tiny"
    ), mock.patch("whisperx.asr_mlx.MlxWhisperPipeline") as mlx_cls, mock.patch.dict(
        sys.modules, modules if modules is not None else _fake_mlx_modules(None)
    ):
        asr.load_model(kwargs.pop("model", "tiny"), device=device, language="en", **kwargs)
    return vad_cls, whisper_cls, mlx_cls


def test_load_model_on_mps_transcribes_with_mlx_and_runs_vad_on_mps():
    import torch

    vad_cls, whisper_cls, mlx_cls = _load_model_on("mps")
    mlx_cls.assert_called_once()
    whisper_cls.assert_not_called()
    assert vad_cls.call_args.args[0] == torch.device("mps")


def test_load_model_on_mps_with_default_options_stays_on_mlx():
    from whisperx.asr_mlx import logger

    with mock.patch.object(logger, "warning") as warning:
        _, whisper_cls, mlx_cls = _load_model_on("mps", asr_options={"temperatures": [0.0], "initial_prompt": "Hi."})
    mlx_cls.assert_called_once()
    whisper_cls.assert_not_called()
    warning.assert_not_called()


@pytest.mark.parametrize(
    "asr_options,named",
    [
        ({"beam_size": 5}, "--beam_size 5 (beam search)"),
        ({"patience": 2.0}, "--patience 2.0 (beam search)"),
        ({"hotwords": "WhisperX"}, "--hotwords (hotword prompting)"),
    ],
)
def test_load_model_on_mps_with_options_mlx_cannot_honor_is_an_error_naming_them(asr_options, named):
    with pytest.raises(ValueError) as exc:
        _load_model_on("mps", asr_options=asr_options)
    assert str(exc.value) == f"Transcribing on mps uses MLX, which does not support {named}. Omit it or use another --device"


def test_load_model_on_mps_names_every_option_mlx_cannot_honor():
    with pytest.raises(ValueError, match=r"--beam_size 5 \(beam search\), --hotwords \(hotword prompting\)\. Omit them"):
        _load_model_on("mps", asr_options={"beam_size": 5, "hotwords": "x"})


def test_load_model_on_mps_keeps_mlx_for_beam_size_1():
    _, _, mlx_cls = _load_model_on("mps", asr_options={"beam_size": 1})
    mlx_cls.assert_called_once()


def test_load_model_on_mps_ignores_speed_options_with_a_warning():
    from whisperx.asr_mlx import logger

    with mock.patch.object(logger, "warning") as warning:
        _, _, mlx_cls = _load_model_on("mps", compute_type="int8", threads=8, device_index=1)
    mlx_cls.assert_called_once()
    warning.assert_called_once_with(
        "MLX transcription ignores --compute_type int8 (MLX runs in float16), --threads 8, --device_index 1"
    )


def test_mlx_pipeline_ignores_batch_size_with_a_warning():
    from whisperx.asr_mlx import MlxWhisperPipeline, logger

    pipeline = MlxWhisperPipeline.__new__(MlxWhisperPipeline)
    pipeline.vad_model, pipeline._vad_params, pipeline.preset_language, pipeline.task = None, {}, "en", "transcribe"
    pipeline.options, pipeline.suppress_numerals = {}, False
    with mock.patch.dict(sys.modules, _fake_mlx_modules(None)), mock.patch(
        "whisperx.asr_mlx.get_vad_segments", return_value=[]
    ), mock.patch.object(logger, "warning") as warning:
        pipeline.transcribe(np.zeros(16000, dtype=np.float32), batch_size=16)
        pipeline.transcribe(np.zeros(16000, dtype=np.float32), batch_size=None)
    warning.assert_called_once_with("MLX transcription ignores --batch_size 16 (it transcribes one VAD chunk at a time)")


def _cli_load_model_call(tmp_path, *flags):
    import whisperx.transcribe as T
    from whisperx.__main__ import cli

    model = mock.MagicMock()
    model.transcribe.return_value = {"segments": [], "language": "en"}
    with mock.patch.object(T, "load_model", return_value=model) as load_model, mock.patch.object(
        T, "load_audio", return_value=np.zeros(16000, dtype=np.float32)
    ), mock.patch.object(T, "get_writer", return_value=lambda *a: None), mock.patch.object(
        sys, "argv", ["whisperx", "audio.wav", "--no_align", "--language", "en", "-o", str(tmp_path), *flags]
    ):
        cli()
    return load_model.call_args.kwargs, model.transcribe.call_args.kwargs


def test_cli_passes_only_the_options_the_user_set(tmp_path):
    load_kwargs, transcribe_kwargs = _cli_load_model_call(tmp_path, "--device", "cpu")
    assert "beam_size" not in load_kwargs["asr_options"] and "patience" not in load_kwargs["asr_options"]
    assert load_kwargs["device_index"] is None and load_kwargs["threads"] is None
    assert transcribe_kwargs["batch_size"] == 8


def test_cli_passes_explicit_beam_search_options(tmp_path):
    load_kwargs, transcribe_kwargs = _cli_load_model_call(
        tmp_path, "--device", "cpu", "--beam_size", "5", "--patience", "2", "--batch_size", "4", "--device_index", "1"
    )
    assert load_kwargs["asr_options"]["beam_size"] == 5 and load_kwargs["asr_options"]["patience"] == 2.0
    assert load_kwargs["device_index"] == 1
    assert transcribe_kwargs["batch_size"] == 4


def test_load_model_on_cpu_still_uses_faster_whisper_with_upstream_defaults():
    _, whisper_cls, mlx_cls = _load_model_on("cpu")
    mlx_cls.assert_not_called()
    assert whisper_cls.call_args.kwargs["device_index"] == 0
    assert whisper_cls.call_args.kwargs["cpu_threads"] == 4


def test_load_model_on_mps_without_mlx_installed_explains_extra_before_loading_vad():
    with pytest.raises(ImportError, match=r"whisperx\[mlx\]"):
        _load_model_on("mps", modules={"mlx": None, "mlx.core": None, "mlx_whisper": None})


def test_load_model_on_mps_with_a_model_that_has_no_mlx_checkpoint_is_an_error():
    with pytest.raises(ValueError, match="No MLX checkpoint for model 'distil-large-v3'"):
        _load_model_on("mps", model="distil-large-v3")


@pytest.mark.parametrize("name,repo", [("turbo", "mlx-community/whisper-large-v3-turbo"), ("mlx-community/x", "mlx-community/x")])
def test_mlx_repo_id_accepts_known_names_and_mlx_community_repos(name, repo):
    from whisperx.asr_mlx import mlx_repo_id

    assert mlx_repo_id(name) == repo


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None
    or not torch.backends.mps.is_available()
    or not shutil.which("say"),
    reason="needs whisperx[mlx] on an Apple Silicon Mac (and macOS `say` for the speech clip)",
)
def test_mps_transcribes_real_speech_with_mlx(tmp_path):
    from whisperx.asr import load_model
    from whisperx.audio import SAMPLE_RATE, load_audio

    path = tmp_path / "speech.aiff"
    subprocess.run(["say", "-o", str(path), "The quick brown fox jumps over the lazy dog."], check=True)
    speech = load_audio(str(path))
    model = load_model("tiny.en", device="mps", language="en")
    result = model.transcribe(speech, batch_size=4)

    assert result["language"] == "en"
    assert result["segments"]
    for segment in result["segments"]:
        assert set(segment) == {"text", "start", "end", "avg_logprob"}
        # VAD frame windows can end a few ms past the audio, as with the faster-whisper backend
        assert 0 <= segment["start"] < segment["end"] <= len(speech) / SAMPLE_RATE + 0.1
    assert "quick brown fox" in " ".join(s["text"] for s in result["segments"]).lower()
