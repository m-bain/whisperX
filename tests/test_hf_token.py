"""Tests for Hugging Face token resolution (CLI arg, file, env var)."""

import os

import pytest

from whisperx.token_utils import HF_TOKEN_ENV_VAR, HFTokenError, resolve_hf_token

SECRET = "hf_SECRETvalue123"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV_VAR, raising=False)


def _write(tmp_path, content, name="token"):
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return str(path)


# --- Precedence -------------------------------------------------------------


def test_cli_token_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "env-token")
    assert resolve_hf_token(hf_token="cli-token") == "cli-token"


def test_token_file_takes_precedence_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "env-token")
    path = _write(tmp_path, "file-token")
    assert resolve_hf_token(hf_token_file=path) == "file-token"


def test_env_var_used_when_no_args(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "env-token")
    assert resolve_hf_token() == "env-token"


def test_returns_none_when_nothing_set():
    assert resolve_hf_token() is None


# --- Normalisation ----------------------------------------------------------


def test_token_file_is_read_and_stripped(tmp_path):
    path = _write(tmp_path, "  file-token\n")
    assert resolve_hf_token(hf_token_file=path) == "file-token"


def test_token_file_path_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write(tmp_path, "file-token", name=".hf_token")
    assert resolve_hf_token(hf_token_file=os.path.join("~", ".hf_token")) == "file-token"


def test_env_var_is_stripped(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "  env-token\n")
    assert resolve_hf_token() == "env-token"


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "   ")
    assert resolve_hf_token() is None


# --- Errors (messages must never contain the token) -------------------------


def _assert_rejected(excinfo):
    assert SECRET not in str(excinfo.value)


def test_both_cli_token_and_file_is_an_error(tmp_path):
    path = _write(tmp_path, "file-token")
    with pytest.raises(HFTokenError):
        resolve_hf_token(hf_token="cli-token", hf_token_file=path)


def test_missing_token_file_is_an_error(tmp_path):
    with pytest.raises(HFTokenError, match="not found"):
        resolve_hf_token(hf_token_file=str(tmp_path / "does-not-exist"))


def test_empty_token_file_is_an_error(tmp_path):
    path = _write(tmp_path, "\n")
    with pytest.raises(HFTokenError, match="empty"):
        resolve_hf_token(hf_token_file=path)


def test_unreadable_token_file_is_an_error(tmp_path, monkeypatch):
    path = _write(tmp_path, SECRET)

    def _deny(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("builtins.open", _deny)
    with pytest.raises(HFTokenError, match="Cannot read") as excinfo:
        resolve_hf_token(hf_token_file=path)
    _assert_rejected(excinfo)


def test_non_utf8_token_file_is_an_error(tmp_path):
    path = _write(tmp_path, b"\xff\xfe" + SECRET.encode())
    with pytest.raises(HFTokenError, match="UTF-8") as excinfo:
        resolve_hf_token(hf_token_file=path)
    _assert_rejected(excinfo)


def test_token_file_with_embedded_newline_is_rejected(tmp_path):
    path = _write(tmp_path, f"{SECRET}\ncomment\n")
    with pytest.raises(HFTokenError, match="whitespace or control") as excinfo:
        resolve_hf_token(hf_token_file=path)
    _assert_rejected(excinfo)


@pytest.mark.parametrize("bad", [f"{SECRET} extra", f"{SECRET}\textra", f"{SECRET}\x00"])
def test_cli_token_with_embedded_whitespace_or_control_is_rejected(bad):
    with pytest.raises(HFTokenError) as excinfo:
        resolve_hf_token(hf_token=bad)
    _assert_rejected(excinfo)


def test_env_token_with_embedded_newline_is_rejected(monkeypatch):
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, f"{SECRET}\nextra")
    with pytest.raises(HFTokenError) as excinfo:
        resolve_hf_token()
    _assert_rejected(excinfo)
