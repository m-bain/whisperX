"""Local transcript selects viewer for WhisperX export libraries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import sqlite3
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


DB_NAME = "viewer.sqlite"
EXPORT_NAME = "selects.csv"


def _stable_id(*parts: object) -> str:
    data = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Request body must be a JSON object")
    return value


def _coerce_float(value: Any, fallback: float | None = None) -> float | None:
    if value is None or value == "":
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Any, fallback: int | None = None) -> int | None:
    if value is None or value == "":
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


class ViewerStore:
    """Indexes a WhisperX `_ai_library` and stores manual review selects."""

    def __init__(self, library_dir: str | Path):
        self.library_dir = Path(library_dir).expanduser().resolve()
        self.manifest_path = self.library_dir / "manifest.csv"
        self.db_path = self.library_dir / DB_NAME
        if not self.library_dir.exists():
            raise FileNotFoundError(f"Library directory does not exist: {self.library_dir}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.csv not found in: {self.library_dir}")
        self._ensure_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    relative_file TEXT NOT NULL,
                    output_dir TEXT,
                    json_path TEXT,
                    status TEXT NOT NULL,
                    language TEXT,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    speaker TEXT,
                    text TEXT NOT NULL,
                    language TEXT,
                    avg_logprob REAL,
                    updated_at REAL NOT NULL,
                    UNIQUE(file_id, segment_index)
                );

                CREATE TABLE IF NOT EXISTS selects (
                    id TEXT PRIMARY KEY,
                    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
                    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    relative_file TEXT NOT NULL,
                    adjusted_start REAL NOT NULL,
                    adjusted_end REAL NOT NULL,
                    text_snapshot TEXT NOT NULL,
                    speaker TEXT,
                    status TEXT NOT NULL DEFAULT 'selected',
                    tags TEXT NOT NULL DEFAULT '[]',
                    theme TEXT,
                    hook_strength INTEGER,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(segment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_segments_file ON segments(file_id);
                CREATE INDEX IF NOT EXISTS idx_segments_text ON segments(text);
                CREATE INDEX IF NOT EXISTS idx_selects_file ON selects(file_id);
                CREATE INDEX IF NOT EXISTS idx_selects_status ON selects(status);
                """
            )

    def rescan(self) -> dict[str, int]:
        rows = self._read_manifest()
        now = time.time()
        seen_files: set[str] = set()
        with self.connect() as conn:
            for row in rows:
                source_path = str(Path(row["file"]).expanduser().resolve())
                file_id = _stable_id(source_path)
                json_path = row.get("json") or ""
                status = row.get("status") or "unknown"
                language = None
                segments: list[dict[str, Any]] = []

                if status == "done" and json_path and Path(json_path).exists():
                    language, segments = self._load_transcript_segments(json_path)

                seen_files.add(file_id)
                conn.execute(
                    """
                    INSERT INTO files (
                        id, source_path, relative_file, output_dir, json_path, status,
                        language, segment_count, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source_path=excluded.source_path,
                        relative_file=excluded.relative_file,
                        output_dir=excluded.output_dir,
                        json_path=excluded.json_path,
                        status=excluded.status,
                        language=excluded.language,
                        segment_count=excluded.segment_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        file_id,
                        source_path,
                        row.get("relative_file") or Path(source_path).name,
                        row.get("output_dir") or "",
                        json_path,
                        status,
                        language,
                        len(segments),
                        now,
                    ),
                )

                for index, segment in enumerate(segments):
                    text = str(segment.get("text") or "").strip()
                    start = float(segment.get("start") or 0.0)
                    end = float(segment.get("end") or start)
                    segment_id = _stable_id(file_id, index)
                    conn.execute(
                        """
                        INSERT INTO segments (
                            id, file_id, segment_index, start, end, speaker, text,
                            language, avg_logprob, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(file_id, segment_index) DO UPDATE SET
                            id=excluded.id,
                            start=excluded.start,
                            end=excluded.end,
                            speaker=excluded.speaker,
                            text=excluded.text,
                            language=excluded.language,
                            avg_logprob=excluded.avg_logprob,
                            updated_at=excluded.updated_at
                        """,
                        (
                            segment_id,
                            file_id,
                            index,
                            start,
                            end,
                            segment.get("speaker"),
                            text,
                            language,
                            _coerce_float(segment.get("avg_logprob")),
                            now,
                        ),
                    )
                conn.execute(
                    "DELETE FROM segments WHERE file_id = ? AND segment_index >= ?",
                    (file_id, len(segments)),
                )

            if seen_files:
                placeholders = ",".join("?" for _ in seen_files)
                conn.execute(f"DELETE FROM files WHERE id NOT IN ({placeholders})", tuple(seen_files))
            else:
                conn.execute("DELETE FROM files")

            conn.commit()

        return self.library_summary()

    def _read_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"file", "relative_file", "output_dir", "json", "status"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"manifest.csv missing columns: {', '.join(missing)}")
            return [dict(row) for row in reader]

    def _load_transcript_segments(self, json_path: str) -> tuple[str | None, list[dict[str, Any]]]:
        with Path(json_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments") if isinstance(payload, dict) else []
        if not isinstance(segments, list):
            segments = []
        return payload.get("language"), [seg for seg in segments if isinstance(seg, dict)]

    def library_summary(self) -> dict[str, Any]:
        with self.connect() as conn:
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM files GROUP BY status ORDER BY status"
            ).fetchall()
            totals = conn.execute(
                """
                SELECT
                    COUNT(*) AS files,
                    COALESCE(SUM(segment_count), 0) AS segments,
                    (SELECT COUNT(*) FROM selects) AS selects
                FROM files
                """
            ).fetchone()
            files = conn.execute(
                """
                SELECT id, source_path, relative_file, output_dir, json_path, status,
                       language, segment_count
                FROM files
                ORDER BY relative_file
                """
            ).fetchall()
        return {
            "library_dir": str(self.library_dir),
            "db_path": str(self.db_path),
            "export_path": str(self.library_dir / EXPORT_NAME),
            "counts": {
                "files": totals["files"],
                "segments": totals["segments"],
                "selects": totals["selects"],
                "statuses": {row["status"]: row["count"] for row in status_rows},
            },
            "files": [dict(row) for row in files],
        }

    def list_segments(self, filters: dict[str, str]) -> dict[str, Any]:
        clauses: list[str] = []
        params: list[Any] = []
        q = (filters.get("q") or "").strip()
        if q:
            clauses.append("(segments.text LIKE ? OR files.relative_file LIKE ? OR segments.speaker LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like])
        file_id = (filters.get("file") or "").strip()
        if file_id:
            clauses.append("segments.file_id = ?")
            params.append(file_id)
        speaker = (filters.get("speaker") or "").strip()
        if speaker:
            clauses.append("segments.speaker = ?")
            params.append(speaker)
        status = (filters.get("status") or "").strip()
        if status:
            clauses.append("selects.status = ?")
            params.append(status)
        theme = (filters.get("theme") or "").strip()
        if theme:
            clauses.append("selects.theme LIKE ?")
            params.append(f"%{theme}%")
        tag = (filters.get("tag") or "").strip()
        if tag:
            clauses.append("selects.tags LIKE ?")
            params.append(f"%{tag}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit = max(1, min(_coerce_int(filters.get("limit"), 200) or 200, 1000))
        offset = max(0, _coerce_int(filters.get("offset"), 0) or 0)
        params.extend([limit, offset])

        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    segments.id, segments.file_id, segments.segment_index, segments.start,
                    segments.end, segments.speaker, segments.text, segments.language,
                    segments.avg_logprob, files.source_path, files.relative_file,
                    selects.id AS select_id, selects.status AS select_status,
                    selects.tags AS select_tags, selects.theme AS select_theme,
                    selects.hook_strength AS select_hook_strength,
                    selects.notes AS select_notes,
                    selects.adjusted_start AS select_adjusted_start,
                    selects.adjusted_end AS select_adjusted_end
                FROM segments
                JOIN files ON files.id = segments.file_id
                LEFT JOIN selects ON selects.segment_id = segments.id
                {where}
                ORDER BY files.relative_file, segments.segment_index
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()

        items = []
        for row in rows:
            item = dict(row)
            item["select_tags"] = _json_loads(item.get("select_tags"), [])
            items.append(item)
        return {"segments": items, "limit": limit, "offset": offset}

    def list_selects(self) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT selects.*, files.json_path
                FROM selects
                JOIN files ON files.id = selects.file_id
                ORDER BY selects.updated_at DESC
                """
            ).fetchall()
        return {"selects": [self._select_row(row) for row in rows]}

    def get_manifest_file(self, file_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()

    def upsert_select(self, payload: dict[str, Any]) -> dict[str, Any]:
        segment_id = str(payload.get("segment_id") or "")
        if not segment_id:
            raise ValueError("segment_id is required")

        with self.connect() as conn:
            segment = conn.execute(
                """
                SELECT segments.*, files.source_path, files.relative_file
                FROM segments
                JOIN files ON files.id = segments.file_id
                WHERE segments.id = ?
                """,
                (segment_id,),
            ).fetchone()
            if segment is None:
                raise KeyError(f"Unknown segment_id: {segment_id}")

            existing = conn.execute("SELECT * FROM selects WHERE segment_id = ?", (segment_id,)).fetchone()
            now = time.time()
            select_id = existing["id"] if existing else _stable_id("select", segment_id)
            adjusted_start = _coerce_float(payload.get("adjusted_start"), segment["start"])
            adjusted_end = _coerce_float(payload.get("adjusted_end"), segment["end"])
            if adjusted_start is None or adjusted_end is None:
                raise ValueError("adjusted_start and adjusted_end must be numbers")
            if adjusted_end < adjusted_start:
                raise ValueError("adjusted_end must be greater than or equal to adjusted_start")

            tags = payload.get("tags")
            if tags is None and existing:
                tags_text = existing["tags"]
            elif isinstance(tags, list):
                tags_text = _json_dumps([str(tag).strip() for tag in tags if str(tag).strip()])
            elif isinstance(tags, str):
                tags_text = _json_dumps([tag.strip() for tag in tags.split(",") if tag.strip()])
            else:
                tags_text = "[]"

            status = str(payload.get("status") or (existing["status"] if existing else "selected"))
            theme = payload.get("theme", existing["theme"] if existing else None)
            notes = payload.get("notes", existing["notes"] if existing else None)
            hook_strength = _coerce_int(
                payload.get("hook_strength"),
                existing["hook_strength"] if existing else None,
            )
            if hook_strength is not None:
                hook_strength = max(1, min(hook_strength, 5))

            conn.execute(
                """
                INSERT INTO selects (
                    id, segment_id, file_id, source_path, relative_file, adjusted_start,
                    adjusted_end, text_snapshot, speaker, status, tags, theme,
                    hook_strength, notes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    adjusted_start=excluded.adjusted_start,
                    adjusted_end=excluded.adjusted_end,
                    text_snapshot=excluded.text_snapshot,
                    speaker=excluded.speaker,
                    status=excluded.status,
                    tags=excluded.tags,
                    theme=excluded.theme,
                    hook_strength=excluded.hook_strength,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (
                    select_id,
                    segment_id,
                    segment["file_id"],
                    segment["source_path"],
                    segment["relative_file"],
                    adjusted_start,
                    adjusted_end,
                    segment["text"],
                    segment["speaker"],
                    status,
                    tags_text,
                    theme,
                    hook_strength,
                    notes,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT selects.*, files.json_path FROM selects JOIN files ON files.id = selects.file_id WHERE selects.id = ?",
                (select_id,),
            ).fetchone()
        return self._select_row(row)

    def delete_select(self, select_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM selects WHERE id = ?", (select_id,))
            conn.commit()

    def export_csv(self) -> Path:
        output_path = self.library_dir / EXPORT_NAME
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT selects.*, files.json_path
                FROM selects
                JOIN files ON files.id = selects.file_id
                ORDER BY selects.relative_file, selects.adjusted_start
                """
            ).fetchall()

        fieldnames = [
            "file",
            "relative_file",
            "start",
            "end",
            "theme",
            "hook_strength",
            "status",
            "tags",
            "speaker",
            "text",
            "notes",
            "json_path",
            "segment_id",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                tags = _json_loads(row["tags"], [])
                writer.writerow(
                    {
                        "file": row["source_path"],
                        "relative_file": row["relative_file"],
                        "start": f"{row['adjusted_start']:.3f}",
                        "end": f"{row['adjusted_end']:.3f}",
                        "theme": row["theme"] or "",
                        "hook_strength": row["hook_strength"] or "",
                        "status": row["status"],
                        "tags": ",".join(tags),
                        "speaker": row["speaker"] or "",
                        "text": row["text_snapshot"],
                        "notes": row["notes"] or "",
                        "json_path": row["json_path"] or "",
                        "segment_id": row["segment_id"],
                    }
                )
        return output_path

    def _select_row(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        value = dict(row)
        value["tags"] = _json_loads(value.get("tags"), [])
        return value


class ViewerApp:
    def __init__(self, store: ViewerStore, rescan_interval: float = 5.0):
        self.store = store
        self.rescan_interval = max(0.0, rescan_interval)
        self._last_rescan = 0.0
        self.rescan(force=True)

    def rescan(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if force or now - self._last_rescan >= self.rescan_interval:
            summary = self.store.rescan()
            self._last_rescan = now
            return summary
        return self.store.library_summary()


class ViewerHandler(BaseHTTPRequestHandler):
    app: ViewerApp
    server_version = "WhisperXViewer/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            if path == "/":
                self._send_html(INDEX_HTML)
            elif path == "/api/library":
                self._send_json(self.app.rescan())
            elif path == "/api/segments":
                self.app.rescan()
                self._send_json(self.app.store.list_segments(query))
            elif path == "/api/selects":
                self.app.rescan()
                self._send_json(self.app.store.list_selects())
            elif path == "/api/export/selects.csv":
                self._send_file(self.app.store.export_csv(), download_name=EXPORT_NAME)
            elif path.startswith("/api/video/"):
                file_id = unquote(path.removeprefix("/api/video/"))
                self._send_video(file_id)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/rescan":
                self._send_json(self.app.rescan(force=True))
            elif path == "/api/selects":
                payload = _read_json_body(self)
                self._send_json({"select": self.app.store.upsert_select(payload)}, status=HTTPStatus.CREATED)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_PATCH(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/selects":
                payload = _read_json_body(self)
                self._send_json({"select": self.app.store.upsert_select(payload)})
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except ValueError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        try:
            path = urlparse(self.path).path
            if path.startswith("/api/selects/"):
                select_id = unquote(path.removeprefix("/api/selects/"))
                self.app.store.delete_select(select_id)
                self._send_json({"deleted": select_id})
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, value: str) -> None:
        raw = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, download_name: str | None = None) -> None:
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{html.escape(download_name)}"')
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_video(self, file_id: str) -> None:
        row = self.app.store.get_manifest_file(file_id)
        if row is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown media file")
            return
        path = Path(row["source_path"])
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Media file is missing")
            return

        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            try:
                unit, value = range_header.split("=", 1)
                if unit.strip() != "bytes":
                    raise ValueError
                start_text, end_text = value.split("-", 1)
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else size - 1
                end = min(end, size - 1)
                if start > end or start < 0:
                    raise ValueError
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WhisperX Selects Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #637083;
      --line: #d9dee7;
      --accent: #146c94;
      --good: #18794e;
      --weak: #a13d2d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    header h1 { margin: 0; font-size: 16px; font-weight: 650; }
    header .counts { color: var(--muted); white-space: nowrap; }
    main {
      display: grid;
      grid-template-columns: 300px minmax(420px, 1fr) 360px;
      gap: 12px;
      padding: 12px;
      height: calc(100vh - 52px);
    }
    aside, section {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .pane-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-weight: 650;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 10px;
      border-bottom: 1px solid var(--line);
    }
    input, select, textarea, button {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
    }
    input, select, textarea { padding: 7px 8px; width: 100%; }
    textarea { min-height: 70px; resize: vertical; }
    button {
      padding: 7px 10px;
      cursor: pointer;
      background: #fdfdfd;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .file-list, .segment-list, .select-list { overflow: auto; height: calc(100% - 43px); }
    .file-row, .segment-row, .select-row {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }
    .file-row:hover, .segment-row:hover, .select-row:hover,
    .segment-row.active { background: #edf6fa; }
    .file-row.done .status { color: var(--good); }
    .file-row.pending .status { color: var(--muted); }
    .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .segment-text { margin-top: 4px; white-space: pre-wrap; }
    .selected-mark { color: var(--accent); font-weight: 700; }
    .preview {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 0;
    }
    video {
      width: 100%;
      max-height: 38vh;
      background: #111;
      display: block;
    }
    .segments-wrap {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .editor {
      padding: 10px;
      display: grid;
      gap: 9px;
      border-bottom: 1px solid var(--line);
    }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .hint { color: var(--muted); font-size: 12px; }
    .empty { padding: 16px; color: var(--muted); }
    @media (max-width: 1000px) {
      main { grid-template-columns: 1fr; height: auto; }
      aside, section { min-height: 360px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>WhisperX Selects Viewer</h1>
    <div class="counts" id="counts">Loading...</div>
  </header>
  <main>
    <aside>
      <div class="pane-title">
        <span>Files</span>
        <button id="rescan">Rescan</button>
      </div>
      <div id="fileList" class="file-list"></div>
    </aside>

    <section class="preview">
      <video id="video" controls preload="metadata"></video>
      <div class="segments-wrap">
        <div class="toolbar">
          <input id="search" placeholder="Search transcript, file, speaker">
          <button id="clearFile">All files</button>
        </div>
        <div id="segmentList" class="segment-list"></div>
      </div>
    </section>

    <aside>
      <div class="pane-title">
        <span>Selects</span>
        <button class="primary" id="exportCsv">Export CSV</button>
      </div>
      <div class="editor">
        <div class="hint" id="focusedHint">Focus a segment to create or edit a select.</div>
        <div class="two">
          <input id="start" placeholder="Start">
          <input id="end" placeholder="End">
        </div>
        <div class="two">
          <select id="status">
            <option value="selected">selected</option>
            <option value="good">good</option>
            <option value="maybe">maybe</option>
            <option value="weak">weak</option>
            <option value="unusable">unusable</option>
          </select>
          <select id="hook">
            <option value="">hook</option>
            <option value="1">1</option><option value="2">2</option>
            <option value="3">3</option><option value="4">4</option>
            <option value="5">5</option>
          </select>
        </div>
        <input id="theme" placeholder="Theme">
        <input id="tags" placeholder="Tags, comma separated">
        <textarea id="notes" placeholder="Notes"></textarea>
        <button class="primary" id="saveSelect">Save select</button>
        <div class="hint">Hotkeys: s save, 1-5 hook, g good, m maybe, x weak, p play/pause, arrows move.</div>
      </div>
      <div id="selectList" class="select-list"></div>
    </aside>
  </main>

  <script>
    const state = { files: [], segments: [], selects: [], file: "", focused: null, focusedIndex: -1 };
    const $ = (id) => document.getElementById(id);

    function fmtTime(value) {
      const total = Math.max(0, Number(value) || 0);
      const minutes = Math.floor(total / 60);
      const seconds = (total % 60).toFixed(1).padStart(4, "0");
      return `${minutes}:${seconds}`;
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {"Content-Type": "application/json", ...(options.headers || {})}
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({error: response.statusText}));
        throw new Error(body.error || response.statusText);
      }
      return response.json();
    }

    async function loadLibrary() {
      const data = await api("/api/library");
      state.files = data.files;
      const statuses = data.counts.statuses || {};
      $("counts").textContent = `${data.counts.files} files · ${data.counts.segments} segments · ${data.counts.selects} selects · ${statuses.done || 0} done`;
      renderFiles();
    }

    async function loadSegments() {
      const params = new URLSearchParams();
      if (state.file) params.set("file", state.file);
      if ($("search").value.trim()) params.set("q", $("search").value.trim());
      params.set("limit", "500");
      const data = await api(`/api/segments?${params}`);
      state.segments = data.segments;
      renderSegments();
    }

    async function loadSelects() {
      const data = await api("/api/selects");
      state.selects = data.selects;
      renderSelects();
    }

    function renderFiles() {
      $("fileList").replaceChildren(...state.files.map(file => {
        const row = document.createElement("div");
        row.className = `file-row ${file.status}`;
        row.onclick = () => { state.file = file.id; loadSegments(); };
        const name = document.createElement("div");
        name.textContent = file.relative_file;
        const meta = document.createElement("div");
        meta.className = "meta";
        const status = document.createElement("span");
        status.className = "status";
        status.textContent = file.status;
        meta.append(status, document.createTextNode(` · ${file.segment_count || 0} segments`));
        row.append(name, meta);
        return row;
      }));
    }

    function renderSegments() {
      if (state.segments.length === 0) {
        $("segmentList").innerHTML = '<div class="empty">No matching segments.</div>';
        return;
      }
      $("segmentList").replaceChildren(...state.segments.map((segment, index) => {
        const row = document.createElement("div");
        row.className = `segment-row ${state.focused?.id === segment.id ? "active" : ""}`;
        row.tabIndex = 0;
        row.onclick = () => focusSegment(segment, index, true);
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${segment.relative_file} · ${fmtTime(segment.start)}-${fmtTime(segment.end)} · ${segment.speaker || "speaker ?"}`;
        const text = document.createElement("div");
        text.className = "segment-text";
        text.textContent = segment.text;
        if (segment.select_id) {
          const mark = document.createElement("span");
          mark.className = "selected-mark";
          mark.textContent = "selected · ";
          meta.prepend(mark);
        }
        row.append(meta, text);
        return row;
      }));
    }

    function renderSelects() {
      if (state.selects.length === 0) {
        $("selectList").innerHTML = '<div class="empty">No selects yet.</div>';
        return;
      }
      $("selectList").replaceChildren(...state.selects.map(select => {
        const row = document.createElement("div");
        row.className = "select-row";
        row.onclick = () => {
          state.file = select.file_id;
          $("search").value = "";
          loadSegments().then(() => {
            const index = state.segments.findIndex(segment => segment.id === select.segment_id);
            if (index >= 0) focusSegment(state.segments[index], index, true);
          });
        };
        const title = document.createElement("div");
        title.textContent = `${select.relative_file} · ${fmtTime(select.adjusted_start)}-${fmtTime(select.adjusted_end)}`;
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${select.status}${select.hook_strength ? " · hook " + select.hook_strength : ""}${select.theme ? " · " + select.theme : ""}`;
        const text = document.createElement("div");
        text.className = "segment-text";
        text.textContent = select.text_snapshot;
        row.append(title, meta, text);
        return row;
      }));
    }

    function focusSegment(segment, index, play) {
      state.focused = segment;
      state.focusedIndex = index;
      $("focusedHint").textContent = `${segment.relative_file} · ${fmtTime(segment.start)}-${fmtTime(segment.end)}`;
      $("start").value = (segment.select_adjusted_start ?? segment.start).toFixed(3);
      $("end").value = (segment.select_adjusted_end ?? segment.end).toFixed(3);
      $("status").value = segment.select_status || "selected";
      $("hook").value = segment.select_hook_strength || "";
      $("theme").value = segment.select_theme || "";
      $("tags").value = (segment.select_tags || []).join(", ");
      $("notes").value = segment.select_notes || "";
      const video = $("video");
      const src = `/api/video/${encodeURIComponent(segment.file_id)}`;
      if (!video.src.endsWith(src)) video.src = src;
      video.currentTime = Number($("start").value) || segment.start;
      if (play) video.play().catch(() => {});
      renderSegments();
    }

    async function saveSelect(extra = {}) {
      if (!state.focused) return;
      const payload = {
        segment_id: state.focused.id,
        adjusted_start: Number($("start").value),
        adjusted_end: Number($("end").value),
        status: $("status").value,
        hook_strength: $("hook").value ? Number($("hook").value) : null,
        theme: $("theme").value,
        tags: $("tags").value,
        notes: $("notes").value,
        ...extra
      };
      await api("/api/selects", {method: "POST", body: JSON.stringify(payload)});
      await Promise.all([loadSegments(), loadSelects(), loadLibrary()]);
    }

    let searchTimer = null;
    $("search").addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(loadSegments, 180);
    });
    $("clearFile").onclick = () => { state.file = ""; loadSegments(); };
    $("rescan").onclick = async () => { await api("/api/rescan", {method: "POST"}); await Promise.all([loadLibrary(), loadSegments(), loadSelects()]); };
    $("saveSelect").onclick = () => saveSelect();
    $("exportCsv").onclick = () => { window.location.href = "/api/export/selects.csv"; };

    document.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        const next = Math.min(state.segments.length - 1, state.focusedIndex + 1);
        if (next >= 0) focusSegment(state.segments[next], next, false);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        const prev = Math.max(0, state.focusedIndex - 1);
        if (prev >= 0) focusSegment(state.segments[prev], prev, false);
      } else if (event.key === "s") {
        event.preventDefault(); saveSelect();
      } else if (event.key >= "1" && event.key <= "5") {
        $("hook").value = event.key; saveSelect({hook_strength: Number(event.key)});
      } else if (event.key === "g") {
        $("status").value = "good"; saveSelect({status: "good"});
      } else if (event.key === "m") {
        $("status").value = "maybe"; saveSelect({status: "maybe"});
      } else if (event.key === "x") {
        $("status").value = "weak"; saveSelect({status: "weak"});
      } else if (event.key === "p") {
        const video = $("video");
        if (video.paused) video.play().catch(() => {}); else video.pause();
      }
    });

    Promise.all([loadLibrary(), loadSegments(), loadSelects()]).catch(error => {
      $("counts").textContent = error.message;
    });
  </script>
</body>
</html>
"""


def make_server(host: str, port: int, app: ViewerApp) -> ThreadingHTTPServer:
    class BoundViewerHandler(ViewerHandler):
        pass

    BoundViewerHandler.app = app
    return ThreadingHTTPServer((host, port), BoundViewerHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local WhisperX transcript selects viewer.")
    parser.add_argument("library", help="Path to a WhisperX _ai_library directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument(
        "--rescan-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between automatic manifest rescans",
    )
    args = parser.parse_args(argv)

    store = ViewerStore(args.library)
    app = ViewerApp(store, rescan_interval=args.rescan_interval)
    server = make_server(args.host, args.port, app)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"WhisperX viewer running at {url}")
    print(f"Library: {store.library_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WhisperX viewer")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
