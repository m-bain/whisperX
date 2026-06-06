"""Backup state machine: every transition of BackupService.state.

States (see app.backup.service.BackupState):
    DISABLED -> UNLINKED -> IDLE <-> DIRTY
    IDLE|DIRTY --backup--> BACKING_UP --ok/raise--> IDLE/ERROR
    IDLE|DIRTY --restore--> RESTORING --ok/raise--> IDLE/ERROR
    ERROR --next op ok--> IDLE ;  LINKED --unlink--> UNLINKED

The transient BACKING_UP / RESTORING states are observed from the main thread
while a gated backend blocks the worker mid-operation.
"""

from __future__ import annotations

import os
import threading

import pytest

from app.backup.backend import StorageBackend
from app.backup.local import LocalFsBackend
from app.backup.service import BackupService, BackupState
from app.store import SessionStore


# --- gated backend: lets a test freeze a put/get to catch transient states ---

class GatedBackend(StorageBackend):
    name = "gated"

    def __init__(self, inner: StorageBackend):
        self.inner = inner
        self.entered = threading.Event()   # set when the gated call is reached
        self.release = threading.Event()   # test sets this to let it proceed
        self.gate_put = False
        self.gate_get = False
        self.linked = True

    def is_linked(self) -> bool:
        return self.linked

    def read_manifest(self):
        return self.inner.read_manifest()

    def write_manifest(self, m):
        return self.inner.write_manifest(m)

    def has_object(self, k):
        return self.inner.has_object(k)

    def put_object(self, k, d):
        if self.gate_put:
            self.entered.set()
            assert self.release.wait(5), "gate never released"
        return self.inner.put_object(k, d)

    def get_object(self, k):
        if self.gate_get:
            self.entered.set()
            assert self.release.wait(5), "gate never released"
        return self.inner.get_object(k)

    def delete_object(self, k):
        return self.inner.delete_object(k)

    def list_objects(self):
        return self.inner.list_objects()


# --- helpers -----------------------------------------------------------------

def _store(tmp_path) -> SessionStore:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return SessionStore(str(data_dir))


def _add_session(store, sid="a", audio=b"audio"):
    store.create(sid, filename=f"{sid}.wav", audio_filename="audio.bin",
                 options={}, model="small")
    sdir = store.session_dir(sid)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "audio.bin"), "wb") as f:
        f.write(audio)
    store.mark_done(sid, language=None, diarized=False, model="small",
                    num_segments=0, duration=0.0)


def _run_async(fn):
    """Run fn() on a thread, capturing any exception. Returns the thread + box."""
    box = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    return t, box


# --- static states -----------------------------------------------------------

def test_disabled_when_no_backend(tmp_path):
    svc = BackupService(_store(tmp_path), None)
    assert svc.state is BackupState.DISABLED
    with pytest.raises(RuntimeError):
        svc.backup_now()


def test_unlinked_when_backend_not_linked(tmp_path):
    backend = GatedBackend(LocalFsBackend(str(tmp_path / "remote")))
    backend.linked = False
    svc = BackupService(_store(tmp_path), backend)
    assert svc.state is BackupState.UNLINKED
    with pytest.raises(RuntimeError, match="not linked"):
        svc.backup_now()


# --- idle / dirty ------------------------------------------------------------

def test_linked_never_pushed_is_dirty_then_idle_after_backup(tmp_path):
    store = _store(tmp_path)
    _add_session(store)
    svc = BackupService(store, LocalFsBackend(str(tmp_path / "remote")))

    # linked but never pushed -> local differs from (empty) remote
    assert svc.state is BackupState.DIRTY
    svc.backup_now()
    assert svc.state is BackupState.IDLE


def test_mutation_moves_idle_to_dirty(tmp_path):
    store = _store(tmp_path)
    _add_session(store, "a")
    svc = BackupService(store, LocalFsBackend(str(tmp_path / "remote")))
    svc.backup_now()
    assert svc.state is BackupState.IDLE

    _add_session(store, "b")  # local change
    assert svc.state is BackupState.DIRTY
    svc.backup_now()
    assert svc.state is BackupState.IDLE


def test_unlink_returns_to_unlinked(tmp_path):
    store = _store(tmp_path)
    _add_session(store)
    backend = GatedBackend(LocalFsBackend(str(tmp_path / "remote")))
    svc = BackupService(store, backend)
    svc.backup_now()
    assert svc.state is BackupState.IDLE

    backend.linked = False  # simulate unlink()
    assert svc.state is BackupState.UNLINKED


# --- transient: BACKING_UP ---------------------------------------------------

def test_backing_up_is_observable_then_idle(tmp_path):
    store = _store(tmp_path)
    _add_session(store)
    backend = GatedBackend(LocalFsBackend(str(tmp_path / "remote")))
    backend.gate_put = True
    svc = BackupService(store, backend)

    t, box = _run_async(svc.backup_now)
    assert backend.entered.wait(5), "backup never reached put_object"
    assert svc.state is BackupState.BACKING_UP
    backend.release.set()
    t.join(5)

    assert "error" not in box, box.get("error")
    assert svc.state is BackupState.IDLE


# --- transient: RESTORING ----------------------------------------------------

def test_restoring_is_observable_then_idle(tmp_path):
    store = _store(tmp_path)
    _add_session(store)
    backend = GatedBackend(LocalFsBackend(str(tmp_path / "remote")))
    svc = BackupService(store, backend)
    svc.backup_now()                      # populate the remote
    assert svc.state is BackupState.IDLE

    backend.gate_get = True
    t, box = _run_async(svc.restore)
    assert backend.entered.wait(5), "restore never reached get_object"
    assert svc.state is BackupState.RESTORING
    backend.release.set()
    t.join(5)

    assert "error" not in box, box.get("error")
    assert svc.state is BackupState.IDLE


# --- ERROR + recovery --------------------------------------------------------

def test_failed_backup_enters_error_then_recovers(tmp_path):
    store = _store(tmp_path)
    _add_session(store)
    inner = LocalFsBackend(str(tmp_path / "remote"))
    svc = BackupService(store, inner)

    boom = lambda _m: (_ for _ in ()).throw(OSError("disk full"))
    original = inner.write_manifest
    inner.write_manifest = boom
    with pytest.raises(OSError, match="disk full"):
        svc.backup_now()
    assert svc.state is BackupState.ERROR
    assert svc.last_error == "disk full"

    # recovery: a successful op clears the error
    inner.write_manifest = original
    svc.backup_now()
    assert svc.state is BackupState.IDLE
    assert svc.last_error is None


# --- guard: rejected ops don't change state ----------------------------------

def test_restore_guard_does_not_change_state(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _add_session(store)
    svc = BackupService(store, LocalFsBackend(str(tmp_path / "remote")))
    svc.backup_now()
    assert svc.state is BackupState.IDLE

    monkeypatch.setattr(store, "has_active_jobs", lambda: True)
    with pytest.raises(RuntimeError, match="jobs are active"):
        svc.restore()
    # guard rejected before any activity transition -> still IDLE
    assert svc.state is BackupState.IDLE
