"""Backup orchestration: snapshot-under-lock, incremental sync, restore, triggers.

All the policy lives here; the backend just moves bytes. Design notes:

* **Consistency.** The DB is copied via ``store.snapshot_db`` which holds the same
  lock every mutation takes, so a backup can't capture a torn write. The copy is
  local + fast; the network upload runs *after* the lock is released, off the
  snapshot. Session artifacts are immutable (audio) or written atomically (edits),
  so they're hashed/uploaded without the store lock.
* **Single-device mirror.** Local is authoritative; the remote is a one-way
  mirror. There is intentionally no multi-device merge. (If multi-device is ever
  wanted, the coarse whole-``sessions.db`` mirror would clobber a peer's
  concurrent metadata; the fix would be decomposing the DB into per-session JSON
  so file-level merge works. Out of scope.)
* **Commit point.** ``backend.write_manifest`` is written last and atomically, so
  a backup only "lands" once every referenced blob is already uploaded.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from app.backup import manifest as mf
from app.backup.backend import StorageBackend

logger = logging.getLogger(__name__)


class BackupState(str, Enum):
    """The single observable state of the backup subsystem.

    Composed from three underlying facts — is a backend configured, is it linked,
    and what is the current activity/dirtiness — into one flat enum so callers (UI,
    tests) have a single source of truth. Transitions:

        DISABLED  --configure backend (construction)--> UNLINKED
        UNLINKED  --link()+assess_link-->  IDLE | DIRTY | CONFLICT
        LINKED    --unlink()-->                        UNLINKED
        CONFLICT  --adopt/overwrite-->                 IDLE
        IDLE      --local mutation-->                  DIRTY
        DIRTY     --backup ok-->                       IDLE
        IDLE|DIRTY--backup_now()-->   BACKING_UP --ok-->IDLE / --raise-->ERROR
        IDLE|DIRTY--restore()-->      RESTORING  --ok-->IDLE / --raise-->ERROR
        ERROR     --next op ok-->                       IDLE
    """
    DISABLED = "disabled"      # no backend configured
    UNLINKED = "unlinked"      # backend configured, not linked
    IDLE = "idle"              # linked, in sync, nothing running
    DIRTY = "dirty"            # linked, local changed since last push
    CONFLICT = "conflict"      # linked, remote differs — awaiting the user's choice
    BACKING_UP = "backing_up"  # a push is in progress
    RESTORING = "restoring"    # a pull/restore is in progress
    ERROR = "error"            # last operation failed (sticky until next success)


# Activity sub-states a running operation sets; folded into BackupState.state.
_ACTIVITY = {BackupState.IDLE, BackupState.BACKING_UP,
             BackupState.RESTORING, BackupState.ERROR}


@dataclass
class RemoteState:
    """What bootstrap-on-link found on the remote, for the caller to act on."""
    exists: bool
    generation: int = 0
    entries: int = 0
    total_size: int = 0
    created_at: str = ""


class LinkOutcome(str, Enum):
    """What linking found, comparing the remote against local data.

    Drives the connect flow: only DIVERGED needs the user to choose; the rest act
    automatically (see :meth:`BackupService.assess_link`).
    """
    FRESH = "fresh"              # no remote backup -> seed the first push
    REMOTE_ONLY = "remote_only"  # remote has data, local empty -> auto-adopt
    IN_SYNC = "in_sync"         # remote == local -> nothing to do
    DIVERGED = "diverged"       # both have data and differ -> ASK the user


@dataclass
class LinkAssessment:
    outcome: LinkOutcome
    remote: RemoteState


@dataclass
class BackupResult:
    uploaded: int          # blobs newly pushed this run
    skipped: int           # blobs already present remotely
    generation: int
    root: str              # merkle root after this backup


class BackupService:
    def __init__(self, store, backend: StorageBackend | None, *,
                 interval: int = 0, gc: bool = True,
                 on_change: Callable[[], None] | None = None):
        self._store = store
        self._backend = backend
        self._interval = interval
        self._gc = gc
        self._on_change = on_change
        self._data_dir = store.data_dir
        self._snapshot_dir = os.path.join(self._data_dir, ".backup")
        self._state_path = os.path.join(self._snapshot_dir, "state.json")
        self._sync_lock = threading.Lock()      # one backup at a time
        self._last_signature: str | None = None  # cheap dirty fingerprint
        self._last_root: str | None = None
        self.last_backup_at: str | None = None    # ISO time of the last successful push
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Activity dimension of the state machine (idle/backing_up/restoring/error).
        self._activity = BackupState.IDLE
        self._activity_lock = threading.Lock()
        # Set on a connect-time conflict (remote differs); pauses periodic auto-backup
        # until the user resolves it via adopt_remote()/overwrite_remote().
        self._awaiting_decision = False
        self.last_error: str | None = None
        self._load_state()  # restore per-device dirty signature + last-backup time

    # --- notify + per-device state persistence -----------------------------
    def _notify(self) -> None:
        """Best-effort fire of the on_change hook (listeners re-read status())."""
        cb = self._on_change
        if cb is None:
            return
        try:
            cb()
        except Exception:  # noqa: BLE001 - a flaky listener must not break a backup
            logger.exception("backup on_change callback failed")

    def _load_state(self) -> None:
        """Load the per-device sidecar (dirty signature + last-backup time).

        Lives in ``.backup/state.json`` which the manifest never includes (build only
        walks ``sessions/`` + the DB snapshot), so it's per-machine and not mirrored.
        Tolerates a missing/corrupt file."""
        try:
            with open(self._state_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        self._last_signature = data.get("last_signature")
        self._last_root = data.get("last_root")
        self.last_backup_at = data.get("last_backup_at")

    def _save_state(self) -> None:
        """Persist the dirty signature + last-backup time (atomic write)."""
        os.makedirs(self._snapshot_dir, exist_ok=True)
        payload = {
            "last_signature": self._last_signature,
            "last_root": self._last_root,
            "last_backup_at": self.last_backup_at,
        }
        tmp = f"{self._state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._state_path)
        except OSError:
            logger.warning("could not persist backup state to %s", self._state_path)

    # --- status ------------------------------------------------------------
    @property
    def backend(self) -> StorageBackend | None:
        return self._backend

    @property
    def interval(self) -> int:
        return self._interval

    def is_linked(self) -> bool:
        return self._backend is not None and self._backend.is_linked()

    def is_dirty(self) -> bool:
        """Whether local state changed since the last successful push (cheap)."""
        return mf.cheap_signature(self._data_dir) != self._last_signature

    @property
    def state(self) -> BackupState:
        """The single flat state (see :class:`BackupState`)."""
        if self._backend is None:
            return BackupState.DISABLED
        if not self.is_linked():
            return BackupState.UNLINKED
        activity = self._activity
        if activity in (BackupState.BACKING_UP, BackupState.RESTORING,
                        BackupState.ERROR):
            return activity
        if self._awaiting_decision:
            return BackupState.CONFLICT
        return BackupState.DIRTY if self.is_dirty() else BackupState.IDLE

    def _set_activity(self, activity: BackupState, error: str | None = None) -> None:
        with self._activity_lock:
            self._activity = activity
            self.last_error = error if activity is BackupState.ERROR else None
        self._notify()

    def _set_awaiting_decision(self, value: bool) -> None:
        if self._awaiting_decision != value:
            self._awaiting_decision = value
            self._notify()

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "linked": self.is_linked(),
            "backend": self._backend.name if self._backend else None,
            "dirty": self.is_dirty() if self.is_linked() else None,
            "last_root": self._last_root,
            "last_backup_at": self.last_backup_at,
            "last_error": self.last_error,
            "interval": self._interval,
        }

    # --- core: push --------------------------------------------------------
    def backup_now(self, *, gc: bool | None = None) -> BackupResult:
        """Snapshot the DB, upload changed blobs, commit a new manifest.

        Serialized by ``_sync_lock``; raises RuntimeError if no backend is linked.
        """
        if not self.is_linked():
            raise RuntimeError("backup backend is not linked")
        gc = self._gc if gc is None else gc
        with self._sync_lock:
            self._set_activity(BackupState.BACKING_UP)
            try:
                result = self._do_backup(gc)
            except Exception as exc:  # noqa: BLE001 - record + re-raise
                self._set_activity(BackupState.ERROR, str(exc))
                raise
            self._set_activity(BackupState.IDLE)
            return result

    def _do_backup(self, gc: bool) -> BackupResult:
        os.makedirs(self._snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(self._snapshot_dir, "snapshot.db")
        # 1. consistent DB copy (holds the store write lock)
        self._store.snapshot_db(snapshot_path)

        # 2. capture a cheap signature BEFORE hashing so a change that lands during
        #    this run leaves us dirty for the next pass (no missed updates).
        signature = mf.cheap_signature(self._data_dir)

        remote = self._backend.read_manifest()
        generation = (remote.generation + 1) if remote else 1

        # 3. build local manifest from the snapshot + artifact tree
        local = mf.build_local_manifest(snapshot_path, self._data_dir,
                                        generation=generation)

        # 4. upload blobs whose content is new remotely
        uploaded = skipped = 0
        for path in mf.changed_paths(local, remote):
            key = local.entries[path].hash
            if self._backend.has_object(key):
                skipped += 1
                continue
            src = (snapshot_path if path == "sessions.db"
                   else os.path.join(self._data_dir, path))
            try:
                with open(src, "rb") as fh:
                    self._backend.put_object(key, fh)
                uploaded += 1
            except FileNotFoundError:
                # File deleted mid-run (e.g. session removed); drop it from this
                # manifest so we don't commit a dangling reference. Next backup
                # reconciles the rest.
                logger.warning("skip vanished file during backup: %s", path)
                local.entries.pop(path, None)

        # 5. commit: write the manifest last (atomic) — this is when it "lands"
        self._backend.write_manifest(local)

        # 6. optional GC of blobs no longer referenced
        if gc:
            self._gc_orphans(local)

        root = mf.merkle_root(local)
        self._last_signature = signature
        self._last_root = root
        self.last_backup_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save_state()
        try:
            os.remove(snapshot_path)
        except OSError:
            pass
        logger.info("backup done: gen=%d uploaded=%d skipped=%d root=%s",
                    generation, uploaded, skipped, root[:12])
        return BackupResult(uploaded, skipped, generation, root)

    def _gc_orphans(self, manifest: mf.Manifest) -> None:
        referenced = manifest.object_keys()
        for key in self._backend.list_objects() - referenced:
            self._backend.delete_object(key)

    # --- core: restore (manual) -------------------------------------------
    def restore(self, *, prune: bool = True) -> int:
        """Pull the remote mirror down to local. Returns the file count restored.

        Maintenance op: refuses to run while jobs are active (would race the DB
        swap). Writes artifacts atomically, swaps ``sessions.db`` via
        ``store.swap_db`` (closes + reopens the connection), and — when ``prune``
        — deletes local session files the manifest no longer lists, so local ends
        up an exact mirror of the remote.
        """
        if not self.is_linked():
            raise RuntimeError("backup backend is not linked")
        if self._store.has_active_jobs():
            raise RuntimeError("cannot restore while transcription jobs are active")
        with self._sync_lock:
            self._set_activity(BackupState.RESTORING)
            try:
                restored = self._do_restore(prune)
            except Exception as exc:  # noqa: BLE001 - record + re-raise
                self._set_activity(BackupState.ERROR, str(exc))
                raise
            self._set_activity(BackupState.IDLE)
            return restored

    def _do_restore(self, prune: bool) -> int:
        remote = self._backend.read_manifest()
        if remote is None:
            raise RuntimeError("remote has no backup to restore")

        restored = 0
        for path, entry in remote.entries.items():
            data = self._backend.get_object(entry.hash)
            if path == "sessions.db":
                continue  # swapped last, after all artifacts are in place
            dest = os.path.join(self._data_dir, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tmp = f"{dest}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            restored += 1

        if prune:
            self._prune_local(remote)

        # DB last: stage then atomic swap + reopen.
        if "sessions.db" in remote.entries:
            os.makedirs(self._snapshot_dir, exist_ok=True)
            staged = os.path.join(self._snapshot_dir, "restore.db")
            with open(staged, "wb") as f:
                f.write(self._backend.get_object(remote.entries["sessions.db"].hash))
            self._store.swap_db(staged)
            restored += 1

        self._last_signature = mf.cheap_signature(self._data_dir)
        self._last_root = mf.merkle_root(remote)
        # Local now mirrors the remote; surface when that backup was made.
        self.last_backup_at = remote.created_at or self.last_backup_at
        self._save_state()
        logger.info("restore done: gen=%d files=%d", remote.generation, restored)
        return restored

    def _prune_local(self, remote: mf.Manifest) -> None:
        """Delete local files under sessions/ not present in the remote manifest."""
        keep = set(remote.entries.keys())
        sessions_root = os.path.join(self._data_dir, "sessions")
        if not os.path.isdir(sessions_root):
            return
        for dirpath, _dirs, files in os.walk(sessions_root, topdown=False):
            for name in files:
                ap = os.path.join(dirpath, name)
                logical = os.path.relpath(ap, self._data_dir).replace(os.sep, "/")
                if logical not in keep:
                    os.remove(ap)
            if not os.listdir(dirpath) and dirpath != sessions_root:
                os.rmdir(dirpath)

    # --- bootstrap on link -------------------------------------------------
    def bootstrap(self) -> RemoteState:
        """Inspect the remote right after linking.

        Returns RemoteState; the caller decides:
          * fresh remote (exists=False)   -> call adopt nothing; push with backup_now()
          * remote has data (exists=True) -> let the user choose:
                adopt_remote()    (load existing -> restore down)
                overwrite_remote()(start fresh   -> push local, GC remote)
        """
        if not self.is_linked():
            raise RuntimeError("backup backend is not linked")
        remote = self._backend.probe()
        if remote is None:
            return RemoteState(exists=False)
        return RemoteState(
            exists=True,
            generation=remote.generation,
            entries=len(remote.entries),
            total_size=sum(e.size for e in remote.entries.values()),
            created_at=remote.created_at,
        )

    def assess_link(self) -> LinkAssessment:
        """Classify a freshly-linked remote against local data (see LinkOutcome).

        Cheap when local is empty (no hashing) or the remote is fresh; only hashes
        the local tree when both sides have data — the genuine conflict check. On
        IN_SYNC it records the current root/signature so the status reads idle
        ("up to date") without a redundant push.
        """
        if not self.is_linked():
            raise RuntimeError("backup backend is not linked")
        remote_m = self._backend.probe()
        if remote_m is None:
            return LinkAssessment(LinkOutcome.FRESH, RemoteState(exists=False))
        remote = RemoteState(
            exists=True,
            generation=remote_m.generation,
            entries=len(remote_m.entries),
            total_size=sum(e.size for e in remote_m.entries.values()),
            created_at=remote_m.created_at,
        )
        if not self._store.list():  # local has no sessions -> nothing to lose
            return LinkAssessment(LinkOutcome.REMOTE_ONLY, remote)
        local_root = self._local_root()
        if local_root == mf.merkle_root(remote_m):
            self._last_root = local_root
            self._last_signature = mf.cheap_signature(self._data_dir)
            self.last_backup_at = remote_m.created_at or self.last_backup_at
            self._save_state()
            self._set_awaiting_decision(False)
            return LinkAssessment(LinkOutcome.IN_SYNC, remote)
        # Real conflict: pause periodic auto-backup until the user picks a side.
        self._set_awaiting_decision(True)
        return LinkAssessment(LinkOutcome.DIVERGED, remote)

    def _local_root(self) -> str:
        """Merkle root of the current local tree (DB snapshot + artifacts)."""
        os.makedirs(self._snapshot_dir, exist_ok=True)
        snap = os.path.join(self._snapshot_dir, "assess.db")
        self._store.snapshot_db(snap)
        try:
            local = mf.build_local_manifest(snap, self._data_dir)
        finally:
            try:
                os.remove(snap)
            except OSError:
                pass
        return mf.merkle_root(local)

    def adopt_remote(self) -> int:
        """Bootstrap choice 'load existing': pull the remote down."""
        self._set_awaiting_decision(False)  # conflict resolved
        return self.restore()

    def overwrite_remote(self) -> BackupResult:
        """Bootstrap choice 'start fresh': push local over the remote (GC old blobs)."""
        self._set_awaiting_decision(False)  # conflict resolved
        return self.backup_now(gc=True)

    # --- periodic trigger --------------------------------------------------
    def start_periodic(self) -> None:
        """Background daemon: every ``interval`` seconds, push if linked + dirty."""
        if self._interval <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._periodic_loop, name="backup-periodic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _periodic_loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                # Skip while a connect-time conflict is unresolved — auto-pushing
                # would silently overwrite a remote the user hasn't chosen to replace.
                if (self.is_linked() and not self._awaiting_decision
                        and self.is_dirty()):
                    self.backup_now()
            except Exception:  # noqa: BLE001 - never let the daemon die
                logger.exception("periodic backup failed")
