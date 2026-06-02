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

from flask import (  # noqa: E402 - load .env first
    Flask,
    abort,
    jsonify,
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

# --- Model bundle: loaded once, in a background thread so the server can boot. ---
_bundle = None
_bundle_lock = threading.Lock()
_bundle_error: str | None = None


def _warm_models() -> None:
    global _bundle, _bundle_error
    try:
        b = pipeline.load_bundle()
        with _bundle_lock:
            _bundle = b
        logger.info("Models ready (diarization %s).", "ON" if b.diarize else "OFF")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model load failed")
        with _bundle_lock:
            _bundle_error = str(exc)


_sessions = SessionStore(DATA_DIR)
_interrupted = _sessions.reconcile_startup()
if _interrupted:
    logger.warning("Marked %d interrupted session(s) as error on startup", _interrupted)


def run_session(session_id: str) -> None:
    """Execute the pipeline for a session and persist results + metadata."""
    row = _sessions.get(session_id)
    if row is None:
        raise RuntimeError(f"session {session_id} disappeared")
    opts = row.get("options") or {}
    audio_path = _sessions.audio_path(session_id)
    result = pipeline.run_job(
        _bundle,
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
        model=pipeline.MODEL_NAME,
        num_segments=result.get("num_segments", len(result.get("segments", []))),
        duration=result.get("duration", 0.0),
    )


_queue = JobQueue(_sessions, run_session)


def models_ready() -> bool:
    with _bundle_lock:
        return _bundle is not None


@app.route("/")
def index():
    return render_template(
        "index.html",
        models_ready=models_ready(),
        bundle_error=_bundle_error,
        diarize_enabled=bool(_bundle and _bundle.diarize),
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
    )
    _queue.submit(session_id)
    return render_template("_status.html", state="queued", job_id=session_id)


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
    return ("", 204)


# Kick off model loading at import time (works under both `flask run` and __main__).
threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), threaded=True)
