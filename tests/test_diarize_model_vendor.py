"""Unit tests for app.diarize_model: version derivation, local resolution, vendoring.

These don't touch the network — `vendor()`'s HF calls are mocked — and they don't
need Flask/Playwright, so they run as part of the normal suite.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app import diarize_model as dm

PREFIX = dm.MODEL_PREFIX


def _make_model_dir(root: Path, sha8: str, vendored_at: str, revision: str | None = None) -> Path:
    d = root / f"{PREFIX}.{sha8}"
    (d / "segmentation").mkdir(parents=True)
    (d / "config.yaml").write_text("dependencies:\n  pyannote.audio: 4.0.0\n")
    (d / "manifest.json").write_text(json.dumps({
        "repo_id": dm.REPO_ID,
        "revision": revision or (sha8 + "0" * (40 - len(sha8))),
        "sha8": sha8,
        "vendored_at": vendored_at,
    }))
    return d


def test_derive_version_parses_name_and_manifest(tmp_path):
    d = _make_model_dir(tmp_path, "abcd1234", "2026-01-02", revision="abcd1234ef" + "0" * 30)
    v = dm.derive_version(d)
    assert v["sha8"] == "abcd1234"
    assert v["revision"].startswith("abcd1234ef")
    assert v["vendored_at"] == "2026-01-02"


def test_derive_version_none():
    assert dm.derive_version(None) is None


def test_resolve_prefers_refreshed_over_bundled(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    data = tmp_path / "data"
    bundled.mkdir()
    data.mkdir()
    _make_model_dir(bundled, "11111111", "2026-01-01")
    refreshed = _make_model_dir(data, "22222222", "2026-01-01")  # same date → data wins

    monkeypatch.setattr(dm, "bundled_root", lambda: bundled)
    monkeypatch.setattr(dm, "data_root", lambda: data)

    resolved = dm.resolve_local_model()
    assert resolved == refreshed
    assert dm.derive_version(resolved)["source"] == "refreshed"


def test_resolve_newest_date_wins_within_root(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    _make_model_dir(bundled, "aaaaaaaa", "2026-01-01")
    newer = _make_model_dir(bundled, "bbbbbbbb", "2026-06-01")

    monkeypatch.setattr(dm, "bundled_root", lambda: bundled)
    monkeypatch.setattr(dm, "data_root", lambda: tmp_path / "nonexistent")

    assert dm.resolve_local_model() == newer


def test_resolve_none_when_nothing_vendored(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "bundled_root", lambda: tmp_path / "no-bundle")
    monkeypatch.setattr(dm, "data_root", lambda: tmp_path / "no-data")
    assert dm.resolve_local_model() is None


def test_vendor_copies_and_writes_manifest(tmp_path, monkeypatch):
    # Fake HF snapshot on disk (what snapshot_download would return).
    snapshot = tmp_path / "snap"
    (snapshot / "embedding").mkdir(parents=True)
    (snapshot / "config.yaml").write_text("dependencies:\n  pyannote.audio: 4.0.0\n")
    (snapshot / "embedding" / "pytorch_model.bin").write_bytes(b"weights")
    (snapshot / "README.md").write_text("docs")  # should be ignored

    full_sha = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda repo_id, revision=None, token=None: str(snapshot)

    class _FakeApi:
        def model_info(self, repo_id, token=None):
            return types.SimpleNamespace(sha=full_sha)

    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    dest_root = tmp_path / "out"
    dest = dm.vendor(token="tok", dest_root=dest_root)

    assert dest.name == f"{PREFIX}.3533c8cf"
    assert (dest / "config.yaml").is_file()
    assert (dest / "embedding" / "pytorch_model.bin").read_bytes() == b"weights"
    assert not (dest / "README.md").exists()  # *.md ignored
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["revision"] == full_sha
    assert manifest["sha8"] == "3533c8cf"
    assert manifest["pyannote_audio"] == "4.0.0"
    assert manifest["vendored_at"]  # date stamped


def test_vendor_idempotent_replaces_existing(tmp_path, monkeypatch):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.yaml").write_text("dependencies:\n  pyannote.audio: 4.0.0\n")

    full_sha = "deadbeef" + "0" * 32
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda repo_id, revision=None, token=None: str(snapshot)
    fake_hub.HfApi = object  # not used: revision pinned below
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    dest_root = tmp_path / "out"
    first = dm.vendor(token=None, dest_root=dest_root, revision=full_sha)
    stale = first / "stale.txt"
    stale.write_text("old")
    second = dm.vendor(token=None, dest_root=dest_root, revision=full_sha)

    assert first == second
    assert not stale.exists()  # dir was replaced, not merged
