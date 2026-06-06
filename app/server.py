"""Flask + htmx frontend for WhisperX (CPU-only, with diarization).

Run:  python -m app.server     (or  flask --app app.server run)
Config: put HF_TOKEN (and optional WHISPERX_* overrides) in app/.env.
        See app/.env.example. In Docker the same file is injected via
        docker-compose `env_file`.
"""

from __future__ import annotations

import logging
import os
import re
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
        value = value.strip()
        if value[:1] not in ('"', "'"):  # strip unquoted inline comment
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        value = value.strip('"').strip("'")
        if key and key not in os.environ:  # don't override an already-set var
            os.environ[key] = value


_load_dotenv()  # before anything reads os.environ

from datetime import datetime, timezone  # noqa: E402

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
from app.sse import Broker, sse_response  # noqa: E402
from app.jobs import JobQueue  # noqa: E402
from app.edits import distinct_speakers, next_speaker_key  # noqa: E402
from app.render import render_markdown, render_transcript, resolve_label  # noqa: E402
from app.store import SessionStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

# Sized for ~4h of WAV: 48 kHz / 16-bit / stereo ≈ 0.7 GB/h → 4h ≈ 2.8 GB, and
# 24-bit stereo ≈ 4.2 GB. Default 5 GB covers those with headroom. Werkzeug
# spools the upload to a temp file (not RAM), so a large cap is safe on the dev
# server. Override via WHISPERX_MAX_UPLOAD_MB.
MAX_UPLOAD_MB = int(os.environ.get("WHISPERX_MAX_UPLOAD_MB", "5000"))
DATA_DIR = os.environ.get("WHISPERX_DATA_DIR", str(Path(__file__).with_name("data")))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _too_large(_e):
    # Flask aborts the request before create_session() runs, so render a clean
    # message into the upload dialog's #dialog-status (htmx innerHTML swap)
    # instead of leaking werkzeug's default HTML / a bare console 413.
    msg = f"File too large. The maximum upload size is {MAX_UPLOAD_MB / 1000:.1f} GB."
    return f'<div class="dialog-error" role="alert">{msg}</div>', 413


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

# SSE pub/sub. Per-session keys carry job progress; one reserved key carries
# global model-load state so the dashboard can react when the background warm
# finishes (flip the "loading models" toast to a success state, refresh dropdowns).
_broker = Broker()
MODELS_CHANNEL = "__models__"
BACKUP_CHANNEL = "__backup__"                 # one-shot OAuth consent flow
BACKUP_STATUS_CHANNEL = "__backup_status__"   # persistent sync-status stream


def _models_event(status: dict) -> dict:
    """SSE payload describing global model-load state.

    Drives the dashboard "loading models" → "ready" toast and refreshes the
    loaded/loading/failed badges on every model <sl-select>. ``models_ready``
    reflects the *active* model (the one a default upload would use)."""
    active = status.get("active")
    active_meta = next((m for m in status["models"] if m["name"] == active), None)
    return {
        "type": "models",
        "models_ready": bool(active_meta and active_meta["loaded"]),
        "active": active,
        "bundle_error": active_meta.get("error") if active_meta else None,
        "diarize_available": status.get("diarize_available"),
        "diarize_error": status.get("diarize_error"),
        "models": status.get("models", []),
    }


def _publish_models(status: dict) -> None:
    """ModelManager on_change hook: broadcast model state to SSE listeners."""
    _broker.publish(MODELS_CHANNEL, _models_event(status))


# Seed the active model + compute device from persisted settings (switches survive
# restart); fall back to the WHISPERX_* defaults if a stored value is no longer valid
# (an unavailable cuda device is downgraded to cpu inside ModelManager).
_manager = pipeline.ModelManager(
    active=_sessions.get_setting("active_model", pipeline.DEFAULT_MODEL),
    device=_sessions.get_setting("device", pipeline.DEFAULT_DEVICE),
    on_change=_publish_models,
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
_backup = backup_pkg.build_service(_sessions, on_change=lambda: _publish_backup())
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
    from app.backup import LinkOutcome, oauth
    try:
        oauth.link_interactive()
    except Exception as exc:  # noqa: BLE001 - surface to caller
        return jsonify({"error": str(exc)}), 400
    if _backup.interval and _backup.is_linked():
        _backup.start_periodic()
    a = _backup.assess_link()
    # Fresh remote -> seed; empty local -> auto-restore; in-sync -> nothing. A
    # DIVERGED remote is left for the client to resolve via /backup/bootstrap/*.
    if a.outcome == LinkOutcome.FRESH:
        _seed_initial_backup()
    elif a.outcome == LinkOutcome.REMOTE_ONLY:
        _seed_initial_restore()
    return jsonify({"linked": True, "outcome": a.outcome.value,
                    "remote": a.remote.__dict__})


@app.post("/backup/unlink")
def backup_unlink():
    from app.backup import oauth
    oauth.unlink()
    return jsonify({"linked": False})


@app.post("/backup/now")
def backup_now():
    if not _backup.is_linked():
        return jsonify({"error": "backup backend is not linked"}), 409
    _run_backup_async()  # progress + result land on /backup/status/events
    return jsonify(_backup.status()), 202


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


# --- Backup UI (partial-rendering, htmx-swapped — mirrors the HF-token card) ---
# These power the Settings "Backup & Restore" card and the onboarding step. They
# reuse the same BackupService as the JSON routes above, but render HTML
# fragments so the UI follows the app's server-rendered-partial convention.
_PROVIDER_LABELS = {"gdrive": "Google Drive", "local": "Local folder"}


def _human_size(n: int) -> str:
    size = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _human_ago(iso: str | None) -> str | None:
    """A loose "5m ago" / "2h ago" for the last-backup time; date for older."""
    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - when).total_seconds()
    if secs < 45:
        return "just now"
    if secs < 3600:
        return f"{round(secs / 60)}m ago"
    if secs < 86400:
        return f"{round(secs / 3600)}h ago"
    return when.astimezone().strftime("%b %-d, %H:%M")


def _backup_ctx(notice: str | None = None, notice_ok: bool = True,
                remote=None, error: str | None = None) -> dict:
    """Template context shared by the Settings card and the onboarding step."""
    status = _backup.status()
    backend = status.get("backend")
    return {
        "backup": status,
        "backup_provider": _PROVIDER_LABELS.get(backend, backend or "Cloud backup"),
        "backup_notice": notice,
        "backup_notice_ok": notice_ok,
        "backup_remote": remote,
        "backup_remote_size": _human_size(remote.total_size) if remote else None,
        "backup_error": error,
        "backup_last_human": _human_ago(status.get("last_backup_at")),
        # User-chosen Drive folder (gdrive only): prefills the connect field and
        # labels the connected card. Stored in the keyring, never in the data dir.
        "backup_folder": backup_pkg.gdrive_folder() if backend == "gdrive" else None,
    }


def _render_backup(template: str, **ctx_overrides):
    return render_template(template, **_backup_ctx(**ctx_overrides))


# --- Live sync status (persistent SSE; mirrors the model-load stream) ----------
# Every BackupService state transition fires on_change -> _publish_backup, which
# re-renders the Settings card and pushes it to the persistent BACKUP_STATUS_CHANNEL
# stream. The browser swaps #backup-card in place, so auto/periodic/manual pushes,
# failures, and "last backup at X" all show live without a reload.
def _backup_status_event() -> dict:
    """Re-render the Settings backup card for the persistent status stream.

    Renders inside an app context (publish may run on a background job thread). In
    the CONFLICT state it re-probes the remote and passes it through so the
    Load/Start-fresh prompt is preserved rather than overwritten by a status push.
    """
    state = _backup.status().get("state")
    remote = None
    if state == "conflict":
        try:
            rs = _backup.bootstrap()
            remote = rs if rs.exists else None
        except Exception:  # noqa: BLE001 - fall back to the plain card
            remote = None
    with app.app_context():
        html = _render_backup("partials/_backup_card.html", remote=remote)
    return {"type": "backup", "state": state, "html": html}


def _publish_backup() -> None:
    """BackupService on_change hook: broadcast the re-rendered card to listeners."""
    try:
        _broker.publish(BACKUP_STATUS_CHANNEL, _backup_status_event())
    except Exception:  # noqa: BLE001 - a render hiccup must not break a backup op
        logger.exception("backup status publish failed")


def _run_backup_async() -> None:
    """Kick a push on a background thread so the request returns immediately and the
    UI shows live "Syncing…" -> "Up to date" via the status stream (on_change)."""
    def run():
        try:
            _backup.backup_now()
        except Exception:  # noqa: BLE001 - state machine records ERROR; surfaced via SSE
            logger.exception("manual backup failed")
    threading.Thread(target=run, name="backup-manual", daemon=True).start()


@app.get("/backup/status/events")
def backup_status_events():
    """Persistent Server-Sent Events stream of backup sync state.

    Emits the current card on connect (correct for late/reconnecting clients), then
    a freshly-rendered card on every state transition. Long-lived — backup state has
    no terminal, so the browser keeps it open until navigation."""
    return sse_response(_broker, BACKUP_STATUS_CHANNEL, initial=_backup_status_event)


# --- Non-blocking OAuth consent ------------------------------------------------
# The Google consent flow runs a loopback HTTP server and blocks until the user
# finishes in the browser, which can take a while. Rather than hold a Flask
# worker (and the originating fetch) open for the whole consent, we run it on a
# background thread and notify the page over SSE (BACKUP_CHANNEL) when it lands.
# The connect routes return a "connecting…" fragment immediately; the page opens
# an EventSource and swaps in the final card on the terminal event.
_backup_link_lock = threading.Lock()
_backup_link = {"active": False, "result": None}  # result: last terminal SSE event


def _start_backup_link(template: str) -> None:
    """Begin the consent flow in the background (idempotent while one runs).

    ``template`` is the partial rendered into the terminal SSE event so the page
    that started the link gets a fragment shaped for its own container.
    """
    with _backup_link_lock:
        if _backup_link["active"]:
            return
        _backup_link["active"] = True
        _backup_link["result"] = None
    threading.Thread(target=_run_backup_link, args=(template,), daemon=True).start()


def _run_backup_link(template: str) -> None:
    from app.backup import LinkOutcome, oauth
    after = None  # "seed" | "restore" — a background op to kick AFTER publishing
    try:
        oauth.link_interactive()
        if _backup.interval and _backup.is_linked():
            _backup.start_periodic()
        a = _backup.assess_link()
        # Only a real conflict (remote has data AND local differs) prompts the user;
        # everything else acts automatically (see LinkOutcome). The adopt/overwrite
        # choice only renders when we pass a truthy `remote`.
        if a.outcome == LinkOutcome.FRESH:
            after = "seed"       # empty remote: push the first backup
        elif a.outcome == LinkOutcome.REMOTE_ONLY:
            after = "restore"    # empty local: pull the existing backup down
        with app.app_context():
            html = _render_backup(
                template,
                remote=a.remote if a.outcome == LinkOutcome.DIVERGED else None,
            )
        event = {"status": "linked", "html": html}
    except Exception as exc:  # noqa: BLE001 - report to the page over SSE
        with app.app_context():
            html = _render_backup(template, notice=str(exc), notice_ok=False)
        event = {"status": "error", "html": html, "message": str(exc)}
    with _backup_link_lock:
        _backup_link["active"] = False
        _backup_link["result"] = event
    _broker.publish(BACKUP_CHANNEL, event)
    # Kick the auto op AFTER publishing "connected" so the UI updates first.
    if after == "seed":
        _seed_initial_backup()
    elif after == "restore":
        _seed_initial_restore()


def _seed_initial_backup() -> None:
    """Push the first backup to a freshly-linked, empty remote — in a background
    thread so neither the link flow nor a request blocks on the upload. (Probing a
    fresh remote on link creates the folder but uploads nothing; without this the
    first push would wait for the periodic loop, up to one interval later.)"""
    def run():
        try:
            result = _backup.backup_now()
            logger.info("initial backup after link: uploaded=%d skipped=%d",
                        result.uploaded, result.skipped)
        except Exception:  # noqa: BLE001 - periodic loop will retry; surface in logs
            logger.exception("initial backup after link failed")
    threading.Thread(target=run, name="backup-initial", daemon=True).start()


def _seed_initial_restore() -> None:
    """Auto-pull a backup onto a freshly-linked, empty device (background thread).

    Only invoked when local has zero sessions (LinkOutcome.REMOTE_ONLY), so there is
    nothing local to lose. Runs off-thread so the connected card flips in first."""
    def run():
        try:
            n = _backup.adopt_remote()
            logger.info("auto-restore after link: restored=%d", n)
        except Exception:  # noqa: BLE001 - surface in logs; user can retry from Settings
            logger.exception("auto-restore after link failed")
    threading.Thread(target=run, name="backup-restore", daemon=True).start()


def _apply_backup_folder(name: str | None) -> None:
    """Persist a user-chosen Drive folder and re-target the live backend.

    Called on connect, before the consent flow runs, so the background bootstrap
    reads the right folder. Blank = keep whatever's stored (default if none).
    """
    name = (name or "").strip()
    if not name:
        return
    try:
        backup_pkg.set_gdrive_folder(name)
    except Exception:  # noqa: BLE001 - keyring missing: still target it this run;
        pass           #   the link itself will surface the real keyring error
    if _backup.backend is not None:
        _backup.backend.set_folder(backup_pkg.gdrive_folder())


def _backup_connecting(card_id: str | None, target: str, swap: str) -> str:
    """Render the 'waiting for sign-in' fragment for a given page container."""
    return render_template(
        "partials/_backup_connecting.html",
        backup_card_id=card_id, backup_target=target, backup_swap=swap,
        **_backup_ctx(),
    )


@app.get("/backup/events")
def backup_events():
    """Server-Sent Events stream for the OAuth consent flow.

    Carries one terminal event (``linked`` / ``error``) with the rendered card
    HTML. A late subscriber (the flow finished before its EventSource opened)
    gets the stored result immediately; otherwise it relays the live event.
    """
    def pending():
        with _backup_link_lock:
            return _backup_link["result"]

    return sse_response(
        _broker,
        BACKUP_CHANNEL,
        pending=pending,
        terminal=lambda e: e.get("status") in ("linked", "error"),
    )


@app.post("/settings/backup/connect")
def settings_backup_connect():
    """Kick off the OAuth consent flow (non-blocking) and return a waiting
    fragment; the final card arrives over SSE (see /backup/events)."""
    if _backup.is_linked():
        return _render_backup("partials/_backup_card.html")
    _apply_backup_folder(request.form.get("backup_folder"))
    _start_backup_link("partials/_backup_card.html")
    return _backup_connecting("backup-card", "#backup-card", "outer")


@app.post("/settings/backup/adopt")
def settings_backup_adopt():
    try:
        n = _backup.adopt_remote()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_card.html",
                              notice=str(exc), notice_ok=False)
    return _render_backup("partials/_backup_card.html",
                          notice=f"Loaded {n} file{'' if n == 1 else 's'} from the backup.")


@app.post("/settings/backup/overwrite")
def settings_backup_overwrite():
    try:
        r = _backup.overwrite_remote()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_card.html",
                              notice=str(exc), notice_ok=False)
    return _render_backup("partials/_backup_card.html",
                          notice=f"Backed up — {r.uploaded} uploaded, {r.skipped} unchanged.")


@app.post("/settings/backup/now")
def settings_backup_now():
    if not _backup.is_linked():
        return _render_backup("partials/_backup_card.html",
                              notice="Backup backend is not linked.", notice_ok=False)
    # Run off-thread so the request returns a "Syncing…" card immediately; the
    # persistent status stream then flips it to "Up to date · last backup …".
    _run_backup_async()
    return _render_backup("partials/_backup_card.html")


@app.post("/settings/backup/restore")
def settings_backup_restore():
    try:
        n = _backup.restore()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_card.html",
                              notice=str(exc), notice_ok=False)
    return _render_backup("partials/_backup_card.html",
                          notice=f"Restored {n} file{'' if n == 1 else 's'} from the backup.")


@app.post("/settings/backup/disconnect")
def settings_backup_disconnect():
    from app.backup import oauth
    oauth.unlink()
    return _render_backup("partials/_backup_card.html",
                          notice="Disconnected. Local data is untouched.")


@app.get("/settings/backup/remote-info")
def settings_backup_remote_info():
    """Detail fragment for the restore modal — probes the remote on open."""
    try:
        remote = _backup.bootstrap()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_remote_info.html", error=str(exc))
    return _render_backup("partials/_backup_remote_info.html", remote=remote)


@app.post("/onboarding/backup/connect")
def onboarding_backup_connect():
    """Non-blocking consent kickoff; the final step fragment arrives over SSE."""
    if _backup.is_linked():
        return _render_backup("partials/_backup_onboarding.html")
    _apply_backup_folder(request.form.get("backup_folder"))
    _start_backup_link("partials/_backup_onboarding.html")
    return _backup_connecting(None, "#ob-backup", "inner")


@app.post("/onboarding/backup/adopt")
def onboarding_backup_adopt():
    try:
        n = _backup.adopt_remote()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_onboarding.html",
                              notice=str(exc), notice_ok=False)
    return _render_backup("partials/_backup_onboarding.html",
                          notice=f"Loaded {n} file{'' if n == 1 else 's'} from your backup.")


@app.post("/onboarding/backup/overwrite")
def onboarding_backup_overwrite():
    try:
        _backup.overwrite_remote()
    except RuntimeError as exc:
        return _render_backup("partials/_backup_onboarding.html",
                              notice=str(exc), notice_ok=False)
    return _render_backup("partials/_backup_onboarding.html",
                          notice="Started a fresh backup on this account.")


@app.post("/onboarding/backup/disconnect")
def onboarding_backup_disconnect():
    from app.backup import oauth
    oauth.unlink()
    return _render_backup("partials/_backup_onboarding.html")


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
        **_backup_ctx(),
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
    {"id": "large-v3-turbo", "name": "Large Turbo", "meta": "809M · 6GB",
     "note": "<b>Near-large accuracy, much faster.</b> Recommended on Apple Silicon (whisper.cpp / Metal). Multilingual."},
]


def _onboarding_size(active: str) -> str:
    """The preselected size card: the active model if it's one of the cards, else
    the platform default (large-v3-turbo on Apple Silicon, small elsewhere)."""
    ids = {s["id"] for s in ONBOARDING_SIZES}
    if active in ids:
        return active
    return pipeline.DEFAULT_MODEL if pipeline.DEFAULT_MODEL in ids else "small"


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
        **_backup_ctx(),
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
        **_backup_ctx(),
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


@app.get("/sessions/<session_id>/speakers")
def list_speakers(session_id: str):
    """The session's speakers (key + resolved label) for the reassign picker."""
    if _sessions.get(session_id) is None:
        abort(404)
    result = _sessions.load_result(session_id) or {}
    segments = _sessions.current_segments(session_id, result.get("segments", []))
    names = _sessions.get_speaker_names(session_id)
    return jsonify([
        {"key": key, "label": resolve_label(key, names)}
        for key in distinct_speakers(segments)
    ])


@app.post("/sessions/<session_id>/turns/<int:turn_index>/speaker")
def reassign_turn(session_id: str, turn_index: int):
    """Reassign one turn to a different speaker (non-destructive overlay edit).

    Send ``speaker=<key>`` to reassign to an existing speaker, or ``name=<name>``
    alone to enroll a brand-new speaker (a fresh key is minted, named, and assigned).
    Returns the re-rendered body (full body, since a reassignment can merge turns and
    shift indices) plus the X-Can-Undo header — same contract as edit_turn.
    """
    if _sessions.get(session_id) is None:
        abort(404)
    speaker = request.form.get("speaker", "").strip()
    name = request.form.get("name", "").strip()
    if not speaker:
        if not name:
            abort(400, "Provide a speaker key or a name for a new speaker.")
        # Enroll a new speaker: mint the next key over everything already in use
        # (segment keys + any named speakers), name it, then reassign to it.
        result = _sessions.load_result(session_id) or {}
        segments = _sessions.current_segments(session_id, result.get("segments", []))
        names = _sessions.get_speaker_names(session_id)
        # Reject duplicates: a new speaker can't reuse an existing speaker's label
        # (custom name or default "Speaker N"), or we'd have two indistinguishable
        # speakers. Compare case-insensitively on the trimmed name.
        taken = {resolve_label(k, names).casefold() for k in distinct_speakers(segments)}
        taken |= {v.casefold() for v in names.values()}
        if name.casefold() in taken:
            abort(409, f"A speaker named {name!r} already exists.")
        existing = set(distinct_speakers(segments)) | set(names)
        speaker = next_speaker_key(existing)
    if name:
        _sessions.set_speaker_name(session_id, speaker, name)
    try:
        _sessions.save_turn_reassign(session_id, turn_index, speaker)
    except IndexError:
        abort(400, "Unknown turn.")
    return _body_response(session_id)


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

    def initial():
        # Read inside the stream (after subscribe) so we can't miss a terminal
        # event that fired in the gap between the existence check and subscribe.
        row = _sessions.get(session_id) or {}
        status = row.get("status")
        if status in ("done", "error"):
            return {"status": status}
        return (_stage_event(row["stage"], row.get("duration")) if row.get("stage")
                else {"status": status or "queued"})

    return sse_response(
        _broker,
        session_id,
        initial=initial,
        terminal=lambda e: e.get("status") in ("done", "error"),
    )


@app.get("/models/events")
def models_events():
    """Server-Sent Events stream of global model-load state.

    Emits the current state immediately on connect (so a late/reconnecting
    client is correct), then a fresh payload each time a model finishes loading
    or the active model / device changes. The client flips the "loading models"
    toast to a success state and refreshes model dropdowns. Long-lived: model
    state has no terminal, so the browser keeps the stream open until navigation.
    """
    return sse_response(_broker, MODELS_CHANNEL, initial=lambda: _models_event(_manager.status()))


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
