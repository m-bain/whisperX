"""Flask + htmx frontend for WhisperX (CPU-only, with diarization).

Run:  python -m app.server     (or  flask --app app.server run)
Config: put HF_TOKEN (and optional WHISPERX_* overrides) in app/.env.
        See app/.env.example. In Docker the same file is injected via
        docker-compose `env_file`.
"""

from __future__ import annotations

import json
import logging
import os
import queue
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
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
)
from werkzeug.utils import secure_filename  # noqa: E402

from app import backup as backup_pkg  # noqa: E402
from app import diarize_model  # noqa: E402
from app import pipeline  # noqa: E402
from app import secret_store  # noqa: E402
from app.events import Broker  # noqa: E402
from app.jobs import JobQueue  # noqa: E402
from app.render import render_markdown, render_transcript, resolve_label  # noqa: E402
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

# Seed the active model + compute device from persisted settings (switches survive
# restart); fall back to the WHISPERX_* defaults if a stored value is no longer valid
# (an unavailable cuda device is downgraded to cpu inside ModelManager).
_manager = pipeline.ModelManager(
    active=_sessions.get_setting("active_model", pipeline.DEFAULT_MODEL),
    device=_sessions.get_setting("device", pipeline.DEFAULT_DEVICE),
)


@app.context_processor
def _inject_device() -> dict:
    """Expose the live compute-device label to every template (sidebar chip)."""
    return {"device_label": pipeline.DEVICE_LABELS.get(_manager.device, _manager.device)}


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
        progress=lambda s: _on_stage(session_id, s),
        on_duration=lambda d: _sessions.mark_duration(session_id, d),
    )
    _sessions.mark_done(
        session_id,
        language=result.get("language"),
        diarized=bool(result.get("diarized")),
        model=model,
        num_segments=result.get("num_segments", len(result.get("segments", []))),
        duration=result.get("duration", 0.0),
    )


_broker = Broker()


def _stage_event(stage: str, duration) -> dict:
    """SSE payload for a stage, with an ETA (seconds) when one can be estimated."""
    event = {"stage": stage}
    eta = pipeline.eta_seconds(stage, duration)
    if eta is not None:
        event["eta"] = round(eta)
    return event


def _on_stage(session_id: str, stage: str) -> None:
    """Persist the live stage (durable, for reconnects) and push it to SSE clients."""
    _sessions.mark_stage(session_id, stage)
    row = _sessions.get(session_id) or {}
    _broker.publish(session_id, _stage_event(stage, row.get("duration")))


_queue = JobQueue(_sessions, run_session, broker=_broker)

# --- Cloud backup: mirror the data dir (DB + artifacts) to a swappable backend.
#     Disabled unless WHISPERX_BACKUP_BACKEND is set. Snapshot-under-lock keeps
#     the DB copy consistent; periodic push runs only when local state changed. ---
_backup = backup_pkg.build_service(_sessions)
if _backup.is_linked():
    _backup.start_periodic()


def models_ready() -> bool:
    return _manager.is_loaded(_manager.active)


# --- Backup endpoints (thin: all logic lives in BackupService) ----------------
@app.get("/backup/status")
def backup_status():
    return jsonify(_backup.status())


@app.post("/backup/link")
def backup_link():
    """Run the OAuth consent flow (loopback) then report what's on the remote."""
    from app.backup import oauth
    try:
        oauth.link_interactive()
    except Exception as exc:  # noqa: BLE001 - surface to caller
        return jsonify({"error": str(exc)}), 400
    if _backup.interval and _backup.is_linked():
        _backup.start_periodic()
    state = _backup.bootstrap()
    return jsonify({"linked": True, "remote": state.__dict__})


@app.post("/backup/unlink")
def backup_unlink():
    from app.backup import oauth
    oauth.unlink()
    return jsonify({"linked": False})


@app.post("/backup/now")
def backup_now():
    try:
        result = _backup.backup_now()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(result.__dict__)


@app.post("/backup/restore")
def backup_restore():
    try:
        restored = _backup.restore()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"restored": restored})


@app.post("/backup/bootstrap/adopt")
def backup_adopt():
    """Bootstrap choice 'load existing': pull the remote down."""
    try:
        restored = _backup.adopt_remote()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"restored": restored})


@app.post("/backup/bootstrap/overwrite")
def backup_overwrite():
    """Bootstrap choice 'start fresh': push local over the remote."""
    try:
        result = _backup.overwrite_remote()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(result.__dict__)


@app.route("/")
def index():
    if _sessions.get_setting("onboarded") != "1":
        return redirect("/onboarding")
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
        diarize_enabled=status["diarize_available"],
        diarize_error=status["diarize_error"],
        models=status,
    )


def _diarize_card_ctx(notice: str | None = None, notice_ok: bool = True) -> dict:
    """Template context for the diarization-model Settings card (and its swaps)."""
    version = diarize_model.derive_version(diarize_model.resolve_local_model())
    return {
        "diarize_version": version,
        "diarize_model_name": diarize_model.REPO_ID,
        "token_set": bool(secret_store.resolve_hf_token()),
        "notice": notice,
        "notice_ok": notice_ok,
    }


@app.get("/settings")
def settings():
    return render_template(
        "settings.html",
        active="settings",
        default_language=_sessions.get_setting("default_language", ""),
        models=_manager.status(),
        **_diarize_card_ctx(),
    )


@app.post("/settings")
def save_settings():
    _sessions.set_setting("default_language", request.form.get("default_language", "").strip())
    return (
        '<span class="frag frag--ok"><sl-icon name="check-circle"></sl-icon> Saved</span>'
    )


# --- Onboarding (first-run setup) -------------------------------------------
# The five model sizes shown in the onboarding "Engine" step (design subset of
# pipeline.WhisperModel; each id is a valid WhisperModel value). Advanced/.en/
# distil variants stay available in Settings.
ONBOARDING_SIZES = [
    {"id": "tiny", "name": "Tiny", "meta": "39M · 1GB",
     "note": "<b>Fastest, lowest accuracy.</b> Best for quick drafts and short, clean recordings."},
    {"id": "base", "name": "Base", "meta": "74M · 1GB",
     "note": "<b>Fast with decent accuracy.</b> A good default for clear single-speaker audio."},
    {"id": "small", "name": "Small", "meta": "244M · 2GB",
     "note": "<b>Balanced speed and accuracy.</b> Handles light background noise well."},
    {"id": "medium", "name": "Medium", "meta": "769M · 5GB",
     "note": "<b>Strong accuracy, slower.</b> Reliable for interviews and accented speech."},
    {"id": "large-v3", "name": "Large-v3", "meta": "1.5B · 10GB",
     "note": "<b>Best accuracy, multilingual.</b> Recommended for research-grade transcripts. Slowest."},
]


def _onboarding_size(active: str) -> str:
    """The preselected size card: the active model if it's one of the five, else 'small'."""
    ids = {s["id"] for s in ONBOARDING_SIZES}
    return active if active in ids else "small"


@app.get("/onboarding")
def onboarding():
    status = _manager.status()
    return render_template(
        "onboarding.html",
        token=secret_store.resolve_hf_token() or "",
        sizes=ONBOARDING_SIZES,
        selected_size=_onboarding_size(status["active"]),
        models=status,
        diarize_model=secret_store.DIARIZE_MODEL,
    )


def _verify_fragment(token: str):
    """Verify a token and return an (ok, html-fragment) pair for the inline notice."""
    ok, detail = secret_store.verify_token(token)
    cls = "frag--ok" if ok else "frag--err"
    icon = "check-circle" if ok else "exclamation-triangle"
    from markupsafe import escape

    html = (f'<span class="frag {cls}"><sl-icon name="{icon}"></sl-icon> '
            f'{escape(detail)}</span>')
    return ok, html


@app.post("/onboarding/verify")
def onboarding_verify():
    """Live-verify the entered token; the Access step gates 'Continue' on the
    X-Token-OK header (and shows the returned notice fragment)."""
    token = request.form.get("token", "").strip()
    ok, html = _verify_fragment(token)
    resp = app.make_response(html)
    resp.headers["X-Token-OK"] = "1" if ok else "0"
    return resp


def _onboarding_store_error(token: str, model: str, exc: Exception):
    """Re-render onboarding on the Access step with a secret-store error."""
    return render_template(
        "onboarding.html",
        token=token, sizes=ONBOARDING_SIZES,
        selected_size=_onboarding_size(model), models=_manager.status(),
        diarize_model=secret_store.DIARIZE_MODEL, store_error=str(exc),
    ), 500


@app.post("/onboarding")
def onboarding_finish():
    """Persist the setup: store the (re-verified) token if one was given, the
    model, and device, mark onboarded, then redirect to the dashboard.

    The token is **optional** — diarization works out of the box from the
    vendored model. A token is stored only to enable refreshing that model from
    Hugging Face. If a token IS supplied it must verify, so we never store junk.
    """
    token = request.form.get("token", "").strip()
    model = request.form.get("model", "").strip()
    device = request.form.get("device", "").strip()

    try:
        model = pipeline.WhisperModel(model).value
    except ValueError:
        abort(400, f"Unknown model: {model}")
    if device not in pipeline.DEVICES:
        abort(400, f"Unknown device: {device}")

    if token:
        ok, _detail = secret_store.verify_token(token)
        if not ok:
            return redirect("/onboarding")
        try:
            secret_store.set_hf_token(token)
        except secret_store.SecretStoreUnavailable as exc:
            return _onboarding_store_error(token, model, exc)

    _sessions.set_setting("active_model", model)
    _sessions.set_setting("device", device)
    _manager.set_active(model)            # background warm
    if device != _manager.device:
        try:
            _manager.set_device(device)   # flush + background warm on new device
        except ValueError:
            pass                          # unavailable device: keep current, dashboard shows it
    _manager.reset_diarize()              # pick up the new token without a restart
    _sessions.set_setting("onboarded", "1")
    return redirect("/")


@app.post("/settings/hf-token")
def settings_hf_token():
    """Verify + store the HF token from Settings; re-render the token card body."""
    token = request.form.get("hf_token", "").strip()
    ok, detail = secret_store.verify_token(token)
    if not ok:
        return render_template("partials/_hf_token.html",
                               token_set=bool(secret_store.resolve_hf_token()),
                               notice=detail, notice_ok=False)
    try:
        secret_store.set_hf_token(token)
    except secret_store.SecretStoreUnavailable as exc:
        return render_template("partials/_hf_token.html",
                               token_set=bool(secret_store.resolve_hf_token()),
                               notice=str(exc), notice_ok=False)
    _manager.reset_diarize()
    return render_template("partials/_hf_token.html", token_set=True,
                           notice="Token saved and verified.", notice_ok=True)


@app.post("/settings/hf-token/clear")
def settings_hf_token_clear():
    """Remove the stored token; diarization falls back to disabled."""
    secret_store.delete_hf_token()
    _manager.reset_diarize()
    token_set = bool(secret_store.resolve_hf_token())  # env override may still apply
    notice = ("Cleared the stored token, but HF_TOKEN is still set in the environment."
              if token_set else "Token cleared. Speaker diarization is now disabled.")
    return render_template("partials/_hf_token.html", token_set=token_set,
                           notice=notice, notice_ok=True)


@app.post("/settings/diarize-model/refresh")
def settings_diarize_refresh():
    """Download the latest pyannote pipeline from HF into the data dir and switch
    to it. Requires a token (the model is gated); re-renders the model card."""
    token = secret_store.resolve_hf_token()
    if not token:
        return render_template(
            "partials/_diarize_model.html",
            **_diarize_card_ctx(
                notice="Add a Hugging Face token above to fetch model updates.",
                notice_ok=False,
            ),
        )
    try:
        dest = diarize_model.vendor(token, dest_root=diarize_model.data_root())
    except Exception as exc:  # noqa: BLE001 - surface download/network errors to the card
        logger.exception("Diarization model refresh failed")
        return render_template(
            "partials/_diarize_model.html",
            **_diarize_card_ctx(notice=f"Refresh failed: {exc}", notice_ok=False),
        )
    _manager.reset_diarize()  # next job re-resolves to the refreshed copy
    sha8 = dest.name.rsplit(".", 1)[-1]
    return render_template(
        "partials/_diarize_model.html",
        **_diarize_card_ctx(notice=f"Refreshed to revision {sha8}.", notice_ok=True),
    )


@app.get("/sessions/<session_id>/view")
def view_session(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    if row["status"] != "done":
        return redirect("/")
    return render_template(
        "transcript.html",
        session=_card(row),
        transcript=_render_body(session_id),
        can_undo=_sessions.edit_history_len(session_id) > 0,
        formats=[f for f in pipeline.OUTPUT_FORMATS
                 if os.path.exists(_sessions.artifact_path(session_id, f))],
    )


def _render_body(session_id: str) -> str:
    """Render the transcript body from the current (edit-overlaid) segments."""
    result = _sessions.load_result(session_id) or {}
    segments = _sessions.current_segments(session_id, result.get("segments", []))
    view = {**result, "segments": segments}
    return render_transcript(view, _sessions.get_speaker_names(session_id))


def _body_response(session_id: str):
    """Transcript body HTML + an X-Can-Undo header so the client can toggle Undo."""
    resp = app.make_response(_render_body(session_id))
    resp.headers["X-Can-Undo"] = "1" if _sessions.edit_history_len(session_id) > 0 else "0"
    return resp


@app.post("/sessions/<session_id>/turns/<int:turn_index>")
def edit_turn(session_id: str, turn_index: int):
    """Edit one turn's text (collapses its segments). Returns the re-rendered body.

    Empty text deletes the turn. The whole body is returned (not just the turn) so a
    deletion that re-merges neighbours and the shifted turn indices stay consistent.
    """
    if _sessions.get(session_id) is None:
        abort(404)
    try:
        _sessions.save_turn_edit(session_id, turn_index, request.form.get("text", ""))
    except IndexError:
        abort(400, "Unknown turn.")
    return _body_response(session_id)


@app.post("/sessions/<session_id>/undo")
def undo_edit(session_id: str):
    """Reverse the most recent turn edit. Returns the re-rendered body."""
    if _sessions.get(session_id) is None:
        abort(404)
    _sessions.undo_turn_edit(session_id)
    return _body_response(session_id)


@app.post("/sessions/<session_id>/rename")
def rename_session(session_id: str):
    """Rename a recording's display title (metadata only). Returns the new name."""
    if _sessions.get(session_id) is None:
        abort(404)
    name = request.form.get("name", "").strip()
    if not name:
        abort(400, "Name cannot be empty.")
    _sessions.rename(session_id, name)
    return name


@app.post("/sessions/<session_id>/speakers")
def rename_speaker(session_id: str):
    """Assign/clear a display name for a diarized speaker (non-destructive).

    Returns the resolved label as plain text so the client can update every
    matching chip in place without re-rendering the transcript.
    """
    if _sessions.get(session_id) is None:
        abort(404)
    speaker = request.form.get("speaker", "").strip()
    if not speaker:
        abort(400, "Missing speaker key.")
    name = request.form.get("name", "").strip()
    _sessions.set_speaker_name(session_id, speaker, name)
    return resolve_label(speaker, {speaker: name} if name else None)


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


@app.post("/device")
def switch_device():
    """Switch the compute device (cpu/cuda): flush + reload all models, persist it.

    Blocked while any job is queued/running so a switch never reloads models out
    from under an in-flight pipeline; the user retries once the queue is idle.
    """
    device = (request.form.get("device") or (request.get_json(silent=True) or {}).get("device") or "").strip()
    if device not in pipeline.DEVICES:
        abort(400, f"Unknown device: {device}")
    if _sessions.has_active_jobs():
        if request.headers.get("HX-Request"):
            return render_template("_device.html", models=_manager.status(), busy=True), 409
        return jsonify({"error": "busy", **_manager.status()}), 409
    try:
        status = _manager.set_device(device)       # flush caches + reload on new device
    except ValueError as exc:
        abort(400, str(exc))
    _sessions.set_setting("device", device)        # survive restart
    if request.headers.get("HX-Request"):
        return render_template("_device.html", models=status)
    return jsonify(status)


@app.get("/sessions/<session_id>/status")
def session_status(session_id: str):
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    status = row["status"]
    if status in ("queued", "running"):
        device_label = pipeline.DEVICE_LABELS.get(_manager.device, _manager.device)
        return render_template("_status.html", state=status, job_id=session_id,
                               device_label=device_label, stage=row.get("stage"))
    if status == "error":
        return render_template("_status.html", state="error", job_id=session_id, error=row["error"])
    # done
    return render_template(
        "_result.html",
        job_id=session_id,
        transcript=_render_body(session_id),
        diarized=bool(row.get("diarized")),
        language=row.get("language"),
        formats=[f for f in pipeline.OUTPUT_FORMATS
                 if os.path.exists(_sessions.artifact_path(session_id, f))],
    )


@app.get("/sessions/<session_id>/events")
def session_events(session_id: str):
    """Server-Sent Events stream of stage/status changes for one session.

    Emits the current state immediately on connect (from the durable row), then
    live deltas via the broker, and closes on a terminal ``status`` event. The
    client reacts to ``done``/``error`` by fetching the final ``/status`` render.
    """
    if _sessions.get(session_id) is None:
        abort(404)

    def stream():
        q = _broker.subscribe(session_id)
        try:
            # Re-read after subscribing so we can't miss a terminal event that
            # fired in the gap between the existence check and the subscribe.
            row = _sessions.get(session_id) or {}
            status = row.get("status")
            if status in ("done", "error"):
                yield _sse({"status": status})
                return
            yield _sse(_stage_event(row["stage"], row.get("duration")) if row.get("stage")
                       else {"status": status or "queued"})
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    yield ": keepalive\n\n"  # comment frame keeps the connection warm
                    continue
                yield _sse(event)
                if event.get("status") in ("done", "error"):
                    return
        finally:
            _broker.unsubscribe(session_id, q)

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


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
    result = _sessions.load_result(session_id)  # segments/words w/ timestamps
    if result is not None:
        # Reflect edits + small-segment coalescing in the exposed data.
        result["segments"] = _sessions.current_segments(session_id, result.get("segments", []))
    row["result"] = result
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


@app.get("/sessions/<session_id>/export.md")
def export_markdown(session_id: str):
    """Markdown transcript of the current (edit-overlaid) segments: title, then
    per-turn speaker tag + timestamp span + text."""
    row = _sessions.get(session_id)
    if row is None:
        abort(404)
    result = _sessions.load_result(session_id) or {}
    segments = _sessions.current_segments(session_id, result.get("segments", []))
    view = {**result, "segments": segments}
    title = row.get("filename") or "Transcript"
    md = render_markdown(view, _sessions.get_speaker_names(session_id), title=title)
    fname = f"{secure_filename(title) or 'transcript'}.md"
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


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
