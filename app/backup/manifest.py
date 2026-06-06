"""Merkle manifest for incremental backups.

A *manifest* is the snapshot of what the local data dir looks like at one point
in time: a map of relative path -> (content hash, size, mtime). Backups are
content-addressed — a file's blob is stored remotely under its hash, so an
unchanged file (e.g. immutable audio) is never re-uploaded, and the manifest is
the merkle root that names the whole tree by its contents.

Layout captured (paths relative to ``data_dir``):
    sessions.db                         (from the *snapshot*, never the live file)
    sessions/<id>/audio.*               (immutable once written)
    sessions/<id>/transcript.json       (+ .edits.json / .srt / .vtt / .txt)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

MANIFEST_VERSION = 1
_CHUNK = 1024 * 1024  # 1 MiB streaming read so large audio doesn't load into RAM


@dataclass
class FileEntry:
    hash: str
    size: int
    mtime: float


@dataclass
class Manifest:
    version: int = MANIFEST_VERSION
    generation: int = 0
    created_at: str = ""
    entries: dict[str, FileEntry] = field(default_factory=dict)

    # --- (de)serialization -------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "generation": self.generation,
                "created_at": self.created_at,
                "entries": {p: asdict(e) for p, e in sorted(self.entries.items())},
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Manifest":
        d = json.loads(raw)
        return cls(
            version=d.get("version", MANIFEST_VERSION),
            generation=d.get("generation", 0),
            created_at=d.get("created_at", ""),
            entries={p: FileEntry(**e) for p, e in d.get("entries", {}).items()},
        )

    def object_keys(self) -> set[str]:
        """The set of content hashes this manifest references (for upload diffs / GC)."""
        return {e.hash for e in self.entries.values()}


def _hash_file(path: str) -> tuple[str, int]:
    """Streamed sha256 + byte size of a file."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _iter_files(root: str):
    """Yield (absolute_path, relative_path) for every file under ``root``."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            ap = os.path.join(dirpath, name)
            yield ap, os.path.relpath(ap, root)


def build_local_manifest(db_snapshot_path: str, data_dir: str,
                         *, generation: int = 0) -> Manifest:
    """Build a manifest from a DB snapshot plus the session-artifact tree.

    ``sessions.db`` is taken from ``db_snapshot_path`` (the consistent copy made
    under the store lock) but recorded under the logical path ``sessions.db`` so
    restore writes it back to the right place. Everything under
    ``data_dir/sessions/`` is included as-is. The ``.backup/`` staging dir and DB
    sidecars are skipped.
    """
    entries: dict[str, FileEntry] = {}

    h, size = _hash_file(db_snapshot_path)
    entries["sessions.db"] = FileEntry(h, size, os.path.getmtime(db_snapshot_path))

    sessions_root = os.path.join(data_dir, "sessions")
    if os.path.isdir(sessions_root):
        for ap, rel in _iter_files(sessions_root):
            logical = os.path.join("sessions", rel).replace(os.sep, "/")
            h, size = _hash_file(ap)
            entries[logical] = FileEntry(h, size, os.path.getmtime(ap))

    return Manifest(
        version=MANIFEST_VERSION,
        generation=generation,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        entries=entries,
    )


def merkle_root(m: Manifest) -> str:
    """A single hash over (sorted path, content hash) pairs.

    Cheap equality check for "is the remote identical to local?" without diffing
    every entry."""
    h = hashlib.sha256()
    for path, entry in sorted(m.entries.items()):
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(entry.hash.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def changed_paths(local: Manifest, remote: Manifest | None) -> list[str]:
    """Paths whose content hash is new vs the remote manifest (need upload)."""
    remote_keys = remote.object_keys() if remote else set()
    return sorted(
        path for path, entry in local.entries.items()
        if entry.hash not in remote_keys
    )


def cheap_signature(data_dir: str) -> str:
    """A fast tree fingerprint from (path, size, mtime) — no content hashing.

    Used for dirty detection in the periodic loop: if this matches the signature
    at the last successful push, nothing changed and we skip the backup."""
    h = hashlib.sha256()
    db = os.path.join(data_dir, "sessions.db")
    targets = [(db, "sessions.db")] if os.path.exists(db) else []
    sessions_root = os.path.join(data_dir, "sessions")
    if os.path.isdir(sessions_root):
        for ap, rel in _iter_files(sessions_root):
            targets.append((ap, os.path.join("sessions", rel).replace(os.sep, "/")))
    for ap, logical in sorted(targets, key=lambda t: t[1]):
        try:
            st = os.stat(ap)
        except OSError:
            continue
        h.update(f"{logical}\0{st.st_size}\0{st.st_mtime_ns}\0".encode("utf-8"))
    return h.hexdigest()
