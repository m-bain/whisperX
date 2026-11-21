"""End-to-end CLI tests for Hugging Face token handling.

These run the real ``whisperx`` CLI entry point but replace model loading,
audio loading and diarization with mocks, so nothing is downloaded. They
check that the resolved token reaches both consumers (ASR and diarization)
and that invalid input fails cleanly without loading models or printing the
token.
"""

import sys
from unittest import mock

import pytest

import whisperx.transcribe as T
from whisperx.__main__ import cli
from whisperx.token_utils import HF_TOKEN_ENV_VAR

SECRET = "hf_SECRETvalue123"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV_VAR, raising=False)


@pytest.fixture
def mocks(monkeypatch):
    model = mock.MagicMock()
    model.transcribe.return_value = {"segments": [], "language": "en"}

    load_model = mock.MagicMock(return_value=model)
    diarization_pipeline = mock.MagicMock()
    diarization_pipeline.return_value.return_value = mock.MagicMock()

    monkeypatch.setattr(T, "load_model", load_model)
    monkeypatch.setattr(T, "load_audio", mock.MagicMock(return_value=mock.MagicMock()))
    monkeypatch.setattr(T, "DiarizationPipeline", diarization_pipeline)
    monkeypatch.setattr(T, "assign_word_speakers", lambda segments, result, emb=None: result)
    monkeypatch.setattr(T, "get_writer", lambda *a, **k: mock.MagicMock())
    return load_model, diarization_pipeline


def _run(monkeypatch, tmp_path, *extra):
    argv = [
        "whisperx", "audio.wav",
        "--device", "cpu",
        "--compute_type", "int8",
        "--no_align",
        "--diarize",
        "--output_dir", str(tmp_path / "out"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli()


def _assert_token_passed(mocks, expected):
    load_model, diarization_pipeline = mocks
    assert load_model.call_args.kwargs["use_auth_token"] == expected
    assert diarization_pipeline.call_args.kwargs["token"] == expected


def test_cli_hf_token(monkeypatch, tmp_path, mocks):
    _run(monkeypatch, tmp_path, "--hf_token", "cli-token")
    _assert_token_passed(mocks, "cli-token")


def test_cli_hf_token_file(monkeypatch, tmp_path, mocks):
    token_file = tmp_path / "token"
    token_file.write_text("file-token\n", encoding="utf-8")
    _run(monkeypatch, tmp_path, "--hf_token_file", str(token_file))
    _assert_token_passed(mocks, "file-token")


def test_cli_env_var(monkeypatch, tmp_path, mocks):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "env-token")
    _run(monkeypatch, tmp_path)
    _assert_token_passed(mocks, "env-token")


def test_cli_no_token_passes_none(monkeypatch, tmp_path, mocks):
    _run(monkeypatch, tmp_path)
    _assert_token_passed(mocks, None)


def _assert_usage_error(excinfo, capsys, mocks):
    load_model, diarization_pipeline = mocks
    assert excinfo.value.code == 2
    load_model.assert_not_called()
    diarization_pipeline.assert_not_called()
    out, err = capsys.readouterr()
    assert SECRET not in out
    assert SECRET not in err
    assert "error:" in err


def test_cli_both_options_is_usage_error(monkeypatch, tmp_path, mocks, capsys):
    token_file = tmp_path / "token"
    token_file.write_text("file-token", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, "--hf_token", SECRET, "--hf_token_file", str(token_file))
    _assert_usage_error(excinfo, capsys, mocks)


def test_cli_missing_token_file_is_usage_error(monkeypatch, tmp_path, mocks, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, "--hf_token_file", str(tmp_path / "missing"))
    _assert_usage_error(excinfo, capsys, mocks)


def test_cli_malformed_token_file_does_not_leak(monkeypatch, tmp_path, mocks, capsys):
    token_file = tmp_path / "token"
    token_file.write_text(f"{SECRET}\ncomment\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, "--hf_token_file", str(token_file))
    _assert_usage_error(excinfo, capsys, mocks)
