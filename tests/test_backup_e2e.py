"""End-to-end backup flows with a mocked cloud service.

The "service" mocked here is the remote (Google Drive): :class:`InMemoryDrive`
implements the exact :class:`StorageBackend` contract that ``app.backup.gdrive``
implements, but keeps objects + the manifest in memory and counts uploads. That
lets us drive the *real* SessionStore + BackupService through full flows without
a network.

Two flows:
  1. Happy restore — machine A backs up; a fresh machine B (separate data dir,
     same remote) bootstraps + restores and ends up an exact replica, down to the
     settings + speaker-name rows carried inside sessions.db.
  2. Initial push then partial update — first backup uploads everything; a later
     edit + new session re-uploads only what changed (unchanged audio is never
     re-sent) and the superseded blob is garbage-collected.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

from app.backup.backend import StorageBackend
from app.backup.service import BackupService, BackupState
from app.backup.manifest import Manifest
from app.store import SessionStore


# --- mocked remote service ---------------------------------------------------

class InMemoryDrive(StorageBackend):
    """A faithful in-memory stand-in for the Drive backend."""

    name = "memdrive"

    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._manifest: str | None = None
        self.put_calls: Counter = Counter()   # hash -> times uploaded
        self.linked = True

    def is_linked(self) -> bool:
        return self.linked

    def read_manifest(self) -> Manifest | None:
        return Manifest.from_json(self._manifest) if self._manifest else None

    def write_manifest(self, manifest: Manifest) -> None:
        self._manifest = manifest.to_json()

    def has_object(self, key: str) -> bool:
        return key in self._objects

    def put_object(self, key: str, data) -> None:
        self.put_calls[key] += 1
        self._objects[key] = data.read()

    def get_object(self, key: str) -> bytes:
        try:
            return self._objects[key]
        except KeyError:
            raise KeyError(key) from None

    def delete_object(self, key: str) -> None:
        self._objects.pop(key, None)

    def list_objects(self) -> set[str]:
        return set(self._objects)


# --- helpers -----------------------------------------------------------------

def _make_store(tmp_path, name: str) -> SessionStore:
    data_dir = tmp_path / name
    data_dir.mkdir()
    return SessionStore(str(data_dir))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _create_session(store, sid, *, audio: bytes, transcript: dict,
                    edits: dict | None = None):
    """Create a realistic session: row + audio + transcript.json (+ edits)."""
    store.create(sid, filename=f"{sid}.wav", audio_filename="audio.bin",
                 options={"language": "en"}, model="small")
    sdir = store.session_dir(sid)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "audio.bin"), "wb") as f:
        f.write(audio)
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump(transcript, f)
    if edits is not None:
        with open(store.edits_path(sid), "w", encoding="utf-8") as f:
            json.dump(edits, f)
    store.mark_done(sid, language="en", diarized=True, model="small",
                    num_segments=len(transcript.get("segments", [])), duration=12.0)


def _snapshot_rows(store):
    """A comparable view of all DB-backed state (sessions + settings + speakers)."""
    return {
        "sessions": store.list(),
        "default_language": store.get_setting("default_language"),
        "speakers_a": store.get_speaker_names("a"),
    }


# --- flow 1: happy restore (machine A -> fresh machine B) --------------------

def test_e2e_happy_restore_replicates_remote_on_fresh_machine(tmp_path):
    remote = InMemoryDrive()

    # --- machine A: populate + back up ---
    store_a = _make_store(tmp_path, "machineA")
    _create_session(store_a, "a", audio=b"AAAA-audio",
                    transcript={"segments": [{"start": 0, "end": 1, "text": "hi"}]},
                    edits={"version": 1, "segments": [], "history": []})
    _create_session(store_a, "b", audio=b"BBBB-audio",
                    transcript={"segments": [{"start": 0, "end": 2, "text": "yo"}]})
    store_a.set_setting("default_language", "en")
    store_a.set_speaker_name("a", "SPEAKER_00", "Alice")

    svc_a = BackupService(store_a, remote, interval=0)
    assert svc_a.state is BackupState.DIRTY      # never pushed
    svc_a.backup_now()
    assert svc_a.state is BackupState.IDLE
    before = _snapshot_rows(store_a)

    # --- machine B: fresh, same remote -> bootstrap + restore ---
    store_b = _make_store(tmp_path, "machineB")
    svc_b = BackupService(store_b, remote, interval=0)

    state = svc_b.bootstrap()
    assert state.exists is True
    assert state.entries >= 5          # db + 2 audio + 2 transcript + 1 edits
    assert state.generation == 1

    assert store_b.list() == []        # genuinely empty before restore
    svc_b.adopt_remote()               # == restore()
    assert svc_b.state is BackupState.IDLE

    # --- B is now an exact replica of A ---
    after = _snapshot_rows(store_b)
    assert after == before                                   # DB rows + settings + speakers
    assert after["default_language"] == "en"
    assert after["speakers_a"] == {"SPEAKER_00": "Alice"}

    # artifacts restored byte-for-byte
    with open(os.path.join(store_b.session_dir("a"), "audio.bin"), "rb") as f:
        assert f.read() == b"AAAA-audio"
    assert store_b.load_result("b")["segments"][0]["text"] == "yo"
    assert store_b.load_edits("a") == {"version": 1, "segments": [], "history": []}

    # restoring an already-in-sync machine is a no-op state-wise
    assert svc_b.is_dirty() is False


# --- flow 2: initial push, then partial update -------------------------------

def test_e2e_initial_push_then_partial_update(tmp_path):
    remote = InMemoryDrive()
    store = _make_store(tmp_path, "machine")

    audio_a = b"AAAA-audio"
    audio_b = b"BBBB-audio"
    transcript_a = {"segments": [{"start": 0, "end": 1, "text": "hi"}]}
    transcript_b_v1 = {"segments": [{"start": 0, "end": 2, "text": "first"}]}
    _create_session(store, "a", audio=audio_a, transcript=transcript_a)
    _create_session(store, "b", audio=audio_b, transcript=transcript_b_v1)

    svc = BackupService(store, remote, interval=0)

    # --- initial push: everything goes up ---
    r1 = svc.backup_now()
    assert r1.generation == 1
    m1 = remote.read_manifest()
    assert set(m1.entries) == {
        "sessions.db",
        "sessions/a/audio.bin", "sessions/a/transcript.json",
        "sessions/b/audio.bin", "sessions/b/transcript.json",
    }
    # each content blob uploaded exactly once
    assert remote.put_calls[_sha(audio_a)] == 1
    assert remote.put_calls[_sha(audio_b)] == 1
    assert remote.put_calls[_sha(json.dumps(transcript_b_v1).encode())] == 1
    old_b_transcript_key = _sha(json.dumps(transcript_b_v1).encode())
    assert old_b_transcript_key in remote.list_objects()
    assert svc.state is BackupState.IDLE

    # --- partial update: edit b's transcript + add a new session c ---
    transcript_b_v2 = {"segments": [{"start": 0, "end": 2, "text": "edited"}]}
    with open(store.result_path("b"), "w", encoding="utf-8") as f:
        json.dump(transcript_b_v2, f)
    store._update("b", num_segments=1)            # touch the DB row too
    audio_c = b"CCCC-audio"
    transcript_c = {"segments": [{"start": 0, "end": 3, "text": "new"}]}
    _create_session(store, "c", audio=audio_c, transcript=transcript_c)

    assert svc.state is BackupState.DIRTY
    r2 = svc.backup_now()
    assert r2.generation == 2

    # only the genuinely-changed blobs were uploaded this round
    new_b_key = _sha(json.dumps(transcript_b_v2).encode())
    assert remote.put_calls[new_b_key] == 1            # b's new transcript
    assert remote.put_calls[_sha(audio_c)] == 1        # new session audio
    assert remote.put_calls[_sha(json.dumps(transcript_c).encode())] == 1
    # unchanged audio for a and b was NEVER re-uploaded
    assert remote.put_calls[_sha(audio_a)] == 1
    assert remote.put_calls[_sha(audio_b)] == 1

    # manifest reflects the new tree; the superseded transcript blob was GC'd
    m2 = remote.read_manifest()
    assert "sessions/c/audio.bin" in m2.entries
    assert m2.entries["sessions/b/transcript.json"].hash == new_b_key
    assert old_b_transcript_key not in remote.list_objects()   # orphan collected

    # a third push with no changes uploads nothing
    r3 = svc.backup_now()
    assert r3.uploaded == 0
    assert svc.state is BackupState.IDLE
