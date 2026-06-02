"""Flask + htmx frontend for WhisperX (CPU-only, with diarization).

Run:  python -m app.server     (or  flask --app app.server run)
Config: put HF_TOKEN (and optional WHISPERX_* overrides) in app/.env.
        See app/.env.example. In Docker the same file is injected via
        docker-compose `env_file`.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from app/.env. Real env vars take precedence."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:  # don't override an already-set var
            os.environ[key] = value


_load_dotenv()  # before anything reads os.environ

from datetime import datetime  # noqa: E402

from flask import (  # noqa: E402 - load .env first
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename  # noqa: E402

from app import pipeline  # noqa: E402
from app.jobs import JobQueue  # noqa: E402
from app.render import render_transcript  # noqa: E402
from app.store import SessionStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

MAX_UPLOAD_MB = int(os.environ.get("WHISPERX_MAX_UPLOAD_MB", "200"))
DATA_DIR = os.environ.get("WHISPERX_DATA_DIR", str(Path(__file__).with_name("data")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# --- View formatting (dashboard cards + transcript header) -------------------
_STATUS_META = {
    "done": ("Done", "chip--ok", True),
    "running": ("Processing", "chip--run", False),
    "queued": ("Queued", "chip--run", False),
    "error": ("Error", "chip--err", False),
}


def _fmt_duration(sec) -> str:
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _fmt_clock(total_sec) -> str:
    total_sec = int(total_sec or 0)
    h, rem = divmod(total_sec, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y · %H:%M")
    except ValueError:
        return iso


def _card(row: dict) -> dict:
    label, chip_class, viewable = _STATUS_META.get(row["status"], ("Unprocessed", "", False))
    if row["status"] == "done":
        sub = f"{row.get('num_segments') or 0} segments transcribed"
        if row.get("language"):
            sub += f" · language {row['language']}"
        sub += "."
    elif row["status"] == "error":
        sub = f"Failed: {row.get('error') or 'unknown error'}"
    else:
        sub = "Awaiting transcription on CPU."
    return {
        "id": row["id"],
        "name": row.get("filename") or "Untitled recording",
        "chip_label": label,
        "chip_class": chip_class,
        "viewable": viewable,
        "dur": _fmt_duration(row.get("duration")),
        "date": _fmt_date(row.get("created_at")),
        "sub": sub,
        "model": row.get("model"),
        "language": row.get("language"),
        "diarized": bool(row.get("diarized")),
        "num_segments": row.get("num_segments") or 0,
        "status": row["status"],
    }


def _summary(rows: list[dict]) -> dict:
    done = [r for r in rows if r["status"] == "done"]
    total_audio = sum(r.get("duration") or 0 for r in rows)
    transcribed = sum(r.get("duration") or 0 for r in done)
    pct = round(len(done) / len(rows) * 100) if rows else 0
    return {
        "count": len(rows),
        "transcribed": _fmt_clock(transcribed),
        "total_audio": _fmt_clock(total_audio),
        "pct": pct,
    }

# --- Models: a manager caches multiple Whisper checkpoints; the active one is
#     warmed in a background thread so the server can boot before it is ready. ---
_sessions = SessionStore(DATA_DIR)
_interrupted = _sessions.reconcile_startup()
if _interrupted:
    logger.warning("Marked %d interrupted session(s) as error on startup", _interrupted)

# Seed the active model from the persisted setting (a switch survives restart);
# falls back to the WHISPERX_MODEL default if the stored value is no longer valid.
_manager = pipeline.ModelManager(active=_sessions.get_setting("active_model", pipeline.DEFAULT_MODEL))


def _warm_models() -> None:
    """Warm the active Whisper model + the shared diarizer in the background."""
    try:
        _manager.load_asr(_manager.active)
        diarize = _manager.ensure_diarize()
        logger.info("Models ready (active=%s, diarization %s).",
                    _manager.active, "ON" if diarize else "OFF")
    except Exception:  # noqa: BLE001 - the error is recorded in manager.status()
        logger.exception("Active model load failed")


def run_session(session_id: str) -> None:
    """Execute the pipeline for a session and persist results + metadata."""
    row = _sessions.get(session_id)
    if row is None:
        raise RuntimeError(f"session {session_id} disappeared")
    opts = row.get("options") or {}
    model = row.get("model") or _manager.active  # the model chosen at upload time
    audio_path = _sessions.audio_path(session_id)
    result = pipeline.run_job(
        _manager.bundle_for(model),  # loads on demand + caches; shares diarizer/align
        audio_path,
        _sessions.session_dir(session_id),
        artifact_basename="transcript",
        language=opts.get("language"),
        min_speakers=opts.get("min_speakers"),
        max_speakers=opts.get("max_speakers"),
    )
    _sessions.mark_done(
        session_id,
        language=result.get("language"),
        diarized=bool(result.get("diarized")),
        model=model,
        num_segments=result.get("num_segments", len(result.get("segments", []))),
        duration=result.get("duration", 0.0),
    )


_queue = JobQueue(_sessions, run_session)


def models_ready() -> bool:
    return _manager.is_loaded(_manager.active)


@app.route("/")
def index():
    rows = _sessions.list()  # newest first
    cards = [_card(r) for r in rows]
    status = _manager.status()
    active_error = next(
        (m["error"] for m in status["models"] if m["name"] == status["active"]), None
    )
    return render_template(
        "index.html",
        featured=cards[0] if cards else None,
        older=cards[1:],
        summary=_summary(rows),
        default_language=_sessions.get_setting("default_language", ""),
        models_ready=models_ready(),
        bundle_error=active_error,
        diarize_enabled=status["diarize"],
        models=status,
    )


@app.get("/settings")
def settings():
    return render_template(
        "settings.html",
        active="settings",
        default_language=_sessions.get_setting("default_language", ""),
        models=_manager.status(),
    )


@app.post("/settings")
def save_settings():
    _sessions.set_setting("default_language", request.form.get("default_language", "").strip())
    return (
        '<span class="frag frag--ok"><sl-icon name="check-circle"></sl-icon> Saved</span>'
    )


@app.get("/sessions/<session_id>/view")
def view_session(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    if row["status"] != "done":
        return redirect("/")
    result = _sessions.load_result(session_id) or {}
    return render_template(
        "transcript.html",
        session=_card(row),
        transcript=render_transcript(result),
        formats=[f for f in pipeline.OUTPUT_FORMATS
                 if os.path.exists(_sessions.artifact_path(session_id, f))],
    )


@app.post("/sessions")
def create_session():
    if not models_ready():
        return render_template("_status.html", state="loading_models", job_id=None), 503

    file = request.files.get("audio")
    if not file or not file.filename:
        abort(400, "No audio file uploaded.")

    def _int(name):
        v = request.form.get(name, "").strip()
        return int(v) if v.isdigit() else None

    # Per-upload model override; defaults to the active model. Reject unknown names.
    requested = request.form.get("model", "").strip()
    if requested:
        try:
            model = pipeline.WhisperModel(requested).value
        except ValueError:
            abort(400, f"Unknown model: {requested}")
    else:
        model = _manager.active

    session_id = uuid.uuid4().hex
    safe_name = secure_filename(file.filename) or "audio"
    ext = os.path.splitext(safe_name)[1] or ".bin"
    audio_filename = f"audio{ext}"

    os.makedirs(_sessions.session_dir(session_id), exist_ok=True)
    file.save(os.path.join(_sessions.session_dir(session_id), audio_filename))

    _sessions.create(
        session_id,
        filename=file.filename,
        audio_filename=audio_filename,
        options={
            "language": request.form.get("language", "").strip() or None,
            "min_speakers": _int("min_speakers"),
            "max_speakers": _int("max_speakers"),
        },
        model=model,
    )
    _queue.submit(session_id)
    return render_template("_status.html", state="queued", job_id=session_id)


@app.get("/models")
def list_models():
    """Available models, which are loaded/loading, and the active one."""
    return jsonify(_manager.status())


@app.post("/models/active")
def switch_model():
    """Dedicated switch endpoint: set + persist the global active model and warm it."""
    model = (request.form.get("model") or (request.get_json(silent=True) or {}).get("model") or "").strip()
    try:
        model = pipeline.WhisperModel(model).value
    except ValueError:
        abort(400, f"Unknown model: {model}")
    status = _manager.set_active(model)         # in-memory switch + background warm
    _sessions.set_setting("active_model", model)  # survive restart
    if request.headers.get("HX-Request"):
        return render_template("_models.html", models=status)
    return jsonify(status)


@app.get("/sessions/<session_id>/status")
def session_status(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    status = row["status"]
    if status in ("queued", "running"):
        return render_template("_status.html", state=status, job_id=session_id)
    if status == "error":
        return render_template("_status.html", state="error", job_id=session_id, error=row["error"])
    # done
    result = _sessions.load_result(session_id) or {}
    return render_template(
        "_result.html",
        job_id=session_id,
        transcript=render_transcript(result),
        diarized=bool(row.get("diarized")),
        language=row.get("language"),
        formats=[f for f in pipeline.OUTPUT_FORMATS
                 if os.path.exists(_sessions.artifact_path(session_id, f))],
    )


@app.get("/sessions")
def list_sessions():
    keys = ("id", "filename", "status", "error", "language", "diarized",
            "model", "num_segments", "duration", "created_at", "updated_at")
    return jsonify([{k: s.get(k) for k in keys} for s in _sessions.list()])


@app.get("/sessions/<session_id>")
def get_session(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    row.pop("audio_filename", None)
    row["result"] = _sessions.load_result(session_id)  # segments/words w/ timestamps
    return jsonify(row)


@app.get("/sessions/<session_id>/audio")
def session_audio(session_id: str):
    path = _sessions.audio_path(session_id)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)


@app.get("/sessions/<session_id>/download/<fmt>")
def download(session_id: str, fmt: str):
    if fmt not in pipeline.OUTPUT_FORMATS:
        abort(404)
    path = _sessions.artifact_path(session_id, fmt)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True)


@app.post("/sessions/<session_id>/delete")
def delete_session(session_id: str):
    if not _sessions.delete(session_id):
        abort(404)
    # htmx swaps the row out on a 200 empty body (it skips 204 No Content).
    return ("", 200)


# Kick off model loading at import time (works under both `flask run` and __main__).
threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), threaded=True)
