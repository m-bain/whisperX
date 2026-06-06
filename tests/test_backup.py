"""Backup engine: manifest diff, snapshot-under-lock, incremental push, bootstrap,
restore round-trip, GC.

Everything runs against the local-filesystem backend (no network, no Google
libs) — the same interface the Drive backend implements.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading

import pytest

from app.backup import manifest as mf
from app.backup.local import LocalFsBackend
from app.backup.service import BackupService, BackupState, LinkOutcome
from app.store import SessionStore


# --- helpers -----------------------------------------------------------------

def _make(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = SessionStore(str(data_dir))
    backend = LocalFsBackend(str(tmp_path / "remote"))
    service = BackupService(store, backend, interval=0, gc=True)
    return store, backend, service


def _add_session(store, sid, audio_bytes=b"audio-data"):
    """Create a session row + a per-session audio artifact on disk."""
    store.create(sid, filename=f"{sid}.wav", audio_filename="audio.bin",
                 options={}, model="small")
    sdir = store.session_dir(sid)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "audio.bin"), "wb") as f:
        f.write(audio_bytes)
    # settle out of 'queued' so it isn't counted as an active job (restore gate)
    store.mark_done(sid, language=None, diarized=False, model="small",
                    num_segments=0, duration=0.0)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- manifest / diff ---------------------------------------------------------

def test_changed_paths_detects_new_and_skips_unchanged(tmp_path):
    store, _backend, service = _make(tmp_path)
    _add_session(store, "a", b"aaa")
    snap = str(tmp_path / "snap.db")
    store.snapshot_db(snap)
    m1 = mf.build_local_manifest(snap, store.data_dir)

    # Same tree -> nothing changed vs itself.
    assert mf.changed_paths(m1, m1) == []

    # Add a file -> only the new path is "changed".
    _add_session(store, "b", b"bbb")
    store.snapshot_db(snap)
    m2 = mf.build_local_manifest(snap, store.data_dir)
    changed = mf.changed_paths(m2, m1)
    assert "sessions/b/audio.bin" in changed
    assert "sessions/a/audio.bin" not in changed  # unchanged, in remote already
    assert mf.merkle_root(m1) != mf.merkle_root(m2)


# --- snapshot under lock -----------------------------------------------------

def test_snapshot_under_lock_is_consistent_during_writes(tmp_path):
    store, _backend, _service = _make(tmp_path)
    _add_session(store, "a")

    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            store.set_setting("counter", str(i))
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for n in range(25):
            dest = str(tmp_path / f"snap-{n}.db")
            store.snapshot_db(dest)  # holds the same lock the writer takes
            con = sqlite3.connect(dest)
            try:
                assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                # the snapshot is a real DB with our table + row
                assert con.execute(
                    "SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            finally:
                con.close()
    finally:
        stop.set()
        t.join()


# --- incremental push --------------------------------------------------------

def test_backup_is_incremental(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a", b"aaa")

    r1 = service.backup_now()
    # first push: db + one artifact
    assert r1.uploaded == 2
    assert backend.read_manifest() is not None

    # add a second session, push again: only db + the new artifact move
    _add_session(store, "b", b"bbb")
    r2 = service.backup_now()
    assert r2.uploaded == 2          # sessions.db (changed) + sessions/b/audio.bin
    # the unchanged audio blob from 'a' is still there and was not re-uploaded
    assert _sha(b"aaa") in backend.list_objects()
    assert r2.generation == 2

    # no changes -> only the db blob differs (it always does after open/close?)
    # a third push with no mutation re-snapshots; db content is identical so its
    # hash matches and nothing is uploaded.
    r3 = service.backup_now()
    assert r3.uploaded == 0


# --- bootstrap ---------------------------------------------------------------

def test_bootstrap_fresh_then_existing(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")

    state = service.bootstrap()
    assert state.exists is False

    service.backup_now()

    # a second service over the same remote sees existing state
    service2 = BackupService(store, backend, interval=0)
    state2 = service2.bootstrap()
    assert state2.exists is True
    assert state2.entries >= 2
    assert state2.generation == 1


# --- assess_link (conflict classification on connect) ------------------------

def test_assess_link_fresh_when_no_remote(tmp_path):
    store, _backend, service = _make(tmp_path)
    _add_session(store, "a")
    a = service.assess_link()
    assert a.outcome == LinkOutcome.FRESH
    assert a.remote.exists is False


def test_assess_link_remote_only_when_local_empty(tmp_path):
    # Build a remote from one store, then assess from a fresh empty store.
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_store = SessionStore(str(empty_dir))
    service2 = BackupService(empty_store, backend, interval=0)
    a = service2.assess_link()
    assert a.outcome == LinkOutcome.REMOTE_ONLY
    assert a.remote.exists is True


def test_assess_link_in_sync_when_unchanged(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    a = service.assess_link()
    assert a.outcome == LinkOutcome.IN_SYNC
    assert service.is_dirty() is False  # recorded root/signature -> "up to date"


def test_assess_link_diverged_when_local_changed(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    _add_session(store, "b", b"new-local-only")  # local now has data the remote lacks
    a = service.assess_link()
    assert a.outcome == LinkOutcome.DIVERGED
    assert a.remote.exists is True


# --- sync state machine (last_backup_at, CONFLICT, sidecar, on_change) --------

def test_backup_sets_last_backup_at_and_idle(tmp_path):
    store, _backend, service = _make(tmp_path)
    _add_session(store, "a")
    assert service.status()["last_backup_at"] is None
    service.backup_now()
    assert service.state == BackupState.IDLE
    assert service.status()["last_backup_at"] is not None


def test_assess_link_diverged_sets_conflict_and_pauses_periodic(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    _add_session(store, "b", b"local-only")  # diverge from the remote
    assert service.assess_link().outcome == LinkOutcome.DIVERGED
    assert service.state == BackupState.CONFLICT
    # The periodic loop guards on this flag, so it won't auto-push during a conflict.
    assert service._awaiting_decision is True


def test_overwrite_clears_conflict(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    _add_session(store, "b", b"local-only")
    service.assess_link()
    assert service.state == BackupState.CONFLICT
    service.overwrite_remote()
    assert service._awaiting_decision is False
    assert service.state in (BackupState.IDLE, BackupState.DIRTY)


def test_adopt_clears_conflict(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    _add_session(store, "b", b"local-only")
    service.assess_link()
    assert service.state == BackupState.CONFLICT
    service.adopt_remote()  # restore down -> in sync with remote
    assert service._awaiting_decision is False
    assert service.state == BackupState.IDLE


def test_sidecar_persists_signature_and_timestamp(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    ts = service.last_backup_at
    assert ts is not None

    # A fresh service over the same data dir loads the sidecar: not dirty, same ts.
    service2 = BackupService(store, backend, interval=0)
    assert service2.last_backup_at == ts
    assert service2.is_dirty() is False
    assert service2.state == BackupState.IDLE


def test_on_change_fires_on_activity_transitions(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store = SessionStore(str(data_dir))
    backend = LocalFsBackend(str(tmp_path / "remote"))
    hits = []
    service = BackupService(store, backend, interval=0,
                            on_change=lambda: hits.append(1))
    _add_session(store, "a")
    service.backup_now()  # BACKING_UP -> IDLE = at least two notifications
    assert len(hits) >= 2


# --- restore round-trip ------------------------------------------------------

def test_restore_brings_back_deleted_session(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a", b"keep-me")
    _add_session(store, "b", b"also")
    service.backup_now()

    # lose session 'a' locally (row + files)
    assert store.delete("a") is True
    assert store.get("a") is None
    assert not os.path.exists(os.path.join(store.session_dir("a"), "audio.bin"))

    restored = service.restore()
    assert restored >= 1

    # DB row and artifact are back
    assert store.get("a") is not None
    with open(os.path.join(store.session_dir("a"), "audio.bin"), "rb") as f:
        assert f.read() == b"keep-me"
    # 'b' untouched
    assert store.get("b") is not None


def test_restore_prunes_local_files_not_in_remote(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()

    # a session that exists locally but not in the remote backup
    _add_session(store, "ghost", b"unsynced")
    ghost_file = os.path.join(store.session_dir("ghost"), "audio.bin")
    assert os.path.exists(ghost_file)

    service.restore(prune=True)
    assert not os.path.exists(ghost_file)


def test_restore_refuses_while_jobs_active(tmp_path, monkeypatch):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a")
    service.backup_now()
    monkeypatch.setattr(store, "has_active_jobs", lambda: True)
    with pytest.raises(RuntimeError, match="jobs are active"):
        service.restore()


# --- GC ----------------------------------------------------------------------

def test_gc_removes_orphaned_objects(tmp_path):
    store, backend, service = _make(tmp_path)
    _add_session(store, "a", b"aaa")
    _add_session(store, "b", b"bbb")
    service.backup_now()
    assert _sha(b"bbb") in backend.list_objects()

    store.delete("b")
    service.backup_now(gc=True)
    assert _sha(b"bbb") not in backend.list_objects()  # orphan collected
    assert _sha(b"aaa") in backend.list_objects()       # still referenced
