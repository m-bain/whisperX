"""Durable session storage: SQLite metadata + per-session files on disk.

A *session* is one uploaded audio file plus its transcription result (segments
and words with timestamps + speakers). Metadata lives in SQLite for fast
listing; the audio, the full result JSON, and the srt/vtt/txt artifacts live
under sessions/<id>/ so they survive restarts.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

# Artifact files share a fixed basename so their paths are reconstructable
# from the session id alone (no absolute paths stored in the DB).
ARTIFACT_BASENAME = "transcript"
RESULT_FILE = f"{ARTIFACT_BASENAME}.json"
# Per-session edit overlay: edited segments + capped delta history. Absent until
# the first edit; the original RESULT_FILE is never mutated.
EDITS_FILE = f"{ARTIFACT_BASENAME}.edits.json"

_COLUMNS = (
    "id", "filename", "audio_filename", "status", "stage", "error", "options",
    "language", "diarized", "model", "num_segments", "duration",
    "created_at", "updated_at",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    filename      TEXT,
    audio_filename TEXT,
    status        TEXT NOT NULL,
    stage         TEXT,
    error         TEXT,
    options       TEXT,
    language      TEXT,
    diarized      INTEGER,
    model         TEXT,
    num_segments  INTEGER,
    duration      REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS speaker_names (
    session_id  TEXT NOT NULL,
    speaker_key TEXT NOT NULL,
    name        TEXT NOT NULL,
    PRIMARY KEY (session_id, speaker_key)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self.sessions_root = os.path.join(self.data_dir, "sessions")
        os.makedirs(self.sessions_root, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            os.path.join(self.data_dir, "sessions.db"), check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add columns missing from older DBs (SQLite has no ADD COLUMN IF NOT EXISTS)."""
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(sessions)")}
        if "stage" not in cols:
            self._db.execute("ALTER TABLE sessions ADD COLUMN stage TEXT")

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "sessions.db")

    # --- backup / restore support --------------------------------------
    def snapshot_db(self, dest_path: str) -> None:
        """Write a WAL-safe, internally-consistent copy of the live DB to
        ``dest_path`` using the SQLite Online Backup API.

        The same ``self._lock`` that every mutation holds is taken here, so a
        snapshot can never capture a half-applied write — even while a job is
        mid-transaction or the WAL has uncheckpointed frames. The copy is local
        and fast; callers upload it after the lock is released.
        """
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with self._lock:
            dst = sqlite3.connect(dest_path)
            try:
                self._db.backup(dst)
            finally:
                dst.close()

    def swap_db(self, new_db_path: str) -> None:
        """Replace the live DB file with ``new_db_path`` and reopen the connection.

        Used by restore. Holds the write lock so no mutation runs across the swap;
        closes the connection (checkpointing WAL), moves the new file into place,
        drops any stale -wal/-shm sidecars, then reopens + re-migrates. Callers
        must ensure no jobs are running (see SessionStore.has_active_jobs)."""
        with self._lock:
            self._db.close()
            target = self.db_path
            for sidecar in (f"{target}-wal", f"{target}-shm"):
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            os.replace(new_db_path, target)
            self._db = sqlite3.connect(target, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            with self._db:
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.executescript(_SCHEMA)
                self._migrate()

    # --- path helpers ---------------------------------------------------
    def session_dir(self, session_id: str) -> str:
        return os.path.join(self.sessions_root, session_id)

    def audio_path(self, session_id: str) -> Optional[str]:
        row = self.get(session_id)
        if not row or not row.get("audio_filename"):
            return None
        return os.path.join(self.session_dir(session_id), row["audio_filename"])

    def artifact_path(self, session_id: str, fmt: str) -> str:
        return os.path.join(self.session_dir(session_id), f"{ARTIFACT_BASENAME}.{fmt}")

    def result_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), RESULT_FILE)

    def load_result(self, session_id: str) -> Optional[dict]:
        path = self.result_path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --- transcript edits (non-destructive overlay; original never mutated) ---
    def edits_path(self, session_id: str) -> str:
        return os.path.join(self.session_dir(session_id), EDITS_FILE)

    def load_edits(self, session_id: str) -> Optional[dict]:
        path = self.edits_path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def current_segments(self, session_id: str, original_segments: list) -> list:
        """The edited segment list if an overlay exists, else the coalesced original.

        With no overlay the original is run through the small-segment coalescer so the
        baseline already satisfies the threshold; the original file is never mutated.
        """
        edits = self.load_edits(session_id)
        if edits and edits.get("segments") is not None:
            return edits["segments"]
        from app.edits import coalesce_segments
        return coalesce_segments(original_segments)

    def edit_history_len(self, session_id: str) -> int:
        edits = self.load_edits(session_id)
        return len(edits["history"]) if edits and edits.get("history") else 0

    def _original_segments(self, session_id: str) -> list:
        return (self.load_result(session_id) or {}).get("segments", [])

    def _baseline_segments(self, session_id: str) -> list:
        """Pristine current state with no overlay: the original, coalesced. Edits build
        on this, so the 'all segments >= threshold' invariant holds from the start and
        is preserved by every collapse (an edited turn is one full-span segment)."""
        from app.edits import coalesce_segments
        return coalesce_segments(self._original_segments(session_id))

    def _write_edits(self, session_id: str, segments: list, history: list) -> None:
        """Atomically write the overlay (tmp + os.replace) so readers never see a
        half-written file."""
        path = self.edits_path(session_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "segments": segments, "history": history},
                      f, ensure_ascii=False)
        os.replace(tmp, path)

    def save_turn_edit(self, session_id: str, turn_index: int, new_text: str) -> list:
        """Apply a turn edit, append the delta (history capped), persist. Returns the
        new segment list. Raises IndexError for an unknown turn."""
        from app.edits import HISTORY_LIMIT, apply_turn_edit
        with self._lock:
            edits = self.load_edits(session_id)
            segments = edits["segments"] if edits else self._baseline_segments(session_id)
            history = list(edits["history"]) if edits else []
            new_segments, delta = apply_turn_edit(segments, turn_index, new_text)
            history.append(delta)
            if len(history) > HISTORY_LIMIT:
                history = history[-HISTORY_LIMIT:]
            self._write_edits(session_id, new_segments, history)
            return new_segments

    def save_turn_reassign(self, session_id: str, turn_index: int,
                           new_speaker: str) -> list:
        """Reassign a turn to ``new_speaker``, append the delta (history capped),
        persist. Returns the new segment list — unchanged (and nothing written) when
        the reassign is a no-op. Raises IndexError for an unknown turn."""
        from app.edits import HISTORY_LIMIT, NoChange, apply_turn_reassign
        with self._lock:
            edits = self.load_edits(session_id)
            segments = edits["segments"] if edits else self._baseline_segments(session_id)
            history = list(edits["history"]) if edits else []
            try:
                new_segments, delta = apply_turn_reassign(segments, turn_index, new_speaker)
            except NoChange:
                return segments
            history.append(delta)
            if len(history) > HISTORY_LIMIT:
                history = history[-HISTORY_LIMIT:]
            self._write_edits(session_id, new_segments, history)
            return new_segments

    def undo_turn_edit(self, session_id: str) -> list:
        """Reverse the most recent edit. Returns the resulting segment list. A no-op
        (returns original) when there is nothing to undo; drops the overlay file once
        fully reverted to the pristine original."""
        from app.edits import undo_last
        with self._lock:
            edits = self.load_edits(session_id)
            baseline = self._baseline_segments(session_id)
            if not edits or not edits.get("history"):
                # Nothing left to undo. Return the live state — which, once the
                # oldest deltas have rolled off the 100-cap, may differ from the
                # pristine baseline (those edits are no longer reversible).
                return edits["segments"] if edits else baseline
            new_segments, new_history = undo_last(edits["segments"], edits["history"])
            if not new_history and new_segments == baseline:
                path = self.edits_path(session_id)
                if os.path.exists(path):
                    os.remove(path)
                return baseline
            self._write_edits(session_id, new_segments, new_history)
            return new_segments

    # --- writes ---------------------------------------------------------
    def create(self, session_id: str, filename: str, audio_filename: str,
               options: dict, model: Optional[str] = None) -> None:
        ts = _now()
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO sessions (id, filename, audio_filename, status, options, "
                "model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (session_id, filename, audio_filename, "queued",
                 json.dumps(options), model, ts, ts),
            )

    def mark_running(self, session_id: str) -> None:
        self._update(session_id, status="running")

    def mark_stage(self, session_id: str, stage: Optional[str]) -> None:
        """Record the pipeline stage in flight (decoding/transcribing/aligning/…)."""
        self._update(session_id, stage=stage)

    def mark_duration(self, session_id: str, duration: float) -> None:
        """Persist the clip length once known (after decode) so ETAs survive reconnects."""
        self._update(session_id, duration=duration)

    def mark_done(self, session_id: str, *, language: Optional[str], diarized: bool,
                  model: str, num_segments: int, duration: float) -> None:
        self._update(
            session_id, status="done", stage=None, error=None, language=language,
            diarized=1 if diarized else 0, model=model,
            num_segments=num_segments, duration=duration,
        )

    def mark_error(self, session_id: str, message: str) -> None:
        self._update(session_id, status="error", stage=None, error=message)

    def rename(self, session_id: str, name: str) -> None:
        """Change a recording's display title (metadata only; audio untouched)."""
        self._update(session_id, filename=name)

    def _update(self, session_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE sessions SET {cols} WHERE id=?",
                (*fields.values(), session_id),
            )

    def delete(self, session_id: str) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            existed = cur.rowcount > 0
            self._db.execute(
                "DELETE FROM speaker_names WHERE session_id=?", (session_id,)
            )
        shutil.rmtree(self.session_dir(session_id), ignore_errors=True)
        return existed

    # --- speaker name overrides (non-destructive; applied at render time) ----
    def get_speaker_names(self, session_id: str) -> dict[str, str]:
        """Map of raw speaker key (e.g. SPEAKER_00) -> user-assigned name."""
        with self._lock:
            rows = self._db.execute(
                "SELECT speaker_key, name FROM speaker_names WHERE session_id=?",
                (session_id,),
            ).fetchall()
        return {r["speaker_key"]: r["name"] for r in rows}

    def set_speaker_name(self, session_id: str, speaker_key: str, name: str) -> None:
        """Upsert a speaker name; a blank name clears the override (revert to default)."""
        name = (name or "").strip()
        with self._lock, self._db:
            if not name:
                self._db.execute(
                    "DELETE FROM speaker_names WHERE session_id=? AND speaker_key=?",
                    (session_id, speaker_key),
                )
                return
            self._db.execute(
                "INSERT INTO speaker_names (session_id, speaker_key, name) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(session_id, speaker_key) DO UPDATE SET name=excluded.name",
                (session_id, speaker_key, name),
            )

    # --- settings (durable key/value, e.g. the global active model) -----
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO settings (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # --- reads ----------------------------------------------------------
    def get(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return _row_to_dict(row)

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # --- settings (global key/value) ------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def has_active_jobs(self) -> bool:
        """Whether any session is queued or running (used to gate device switches)."""
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM sessions WHERE status IN ('queued','running') LIMIT 1"
            ).fetchone()
        return row is not None

    # --- lifecycle ------------------------------------------------------
    def reconcile_startup(self) -> list[str]:
        """Reset sessions left mid-flight by a crash/restart; return IDs to requeue."""
        with self._lock, self._db:
            rows = self._db.execute(
                "SELECT id FROM sessions WHERE status IN ('queued','running')"
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._db.execute(
                    "UPDATE sessions SET status='queued', stage=NULL, error=NULL, "
                    "updated_at=? WHERE status IN ('queued','running')",
                    (_now(),),
                )
            return ids


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    d = {k: row[k] for k in row.keys()}
    if d.get("options"):
        try:
            d["options"] = json.loads(d["options"])
        except (ValueError, TypeError):
            d["options"] = {}
    if d.get("diarized") is not None:
        d["diarized"] = bool(d["diarized"])
    return d
