"""Server-side dispatch for the backup *connect* flow.

`assess_link()` (tested in test_backup.py) only *classifies* the remote-vs-local
situation; this file asserts the glue in ``app.server`` that *acts* on each
:class:`LinkOutcome`:

    FRESH       -> seed the first push           (_seed_initial_backup)
    REMOTE_ONLY -> auto-restore the empty device (_seed_initial_restore)
    IN_SYNC     -> nothing
    DIVERGED    -> render the Load/Start-fresh prompt (remote passed to the card)

Both entry points are covered: the background SSE flow ``_run_backup_link`` and
the JSON endpoint ``POST /backup/link``.

The server has import-time side effects (a model warm-up thread, a SessionStore);
we point the data dir at a tmp dir and neutralize model loading before importing
it, mirroring test_onboarding_e2e.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("flask", reason="app dep; see app/requirements.txt")
pytest.importorskip("keyring", reason="app dep; see app/requirements.txt")

from app.backup.service import LinkAssessment, LinkOutcome, RemoteState  # noqa: E402


@pytest.fixture(scope="module")
def server_mod():
    """Import ``app.server`` once, with models + data dir neutralized."""
    from app import pipeline
    pipeline.ModelManager.load_asr = lambda self, name: object()
    pipeline.ModelManager.ensure_diarize = lambda self: None

    tmp = tempfile.mkdtemp(prefix="backup-server-test-")
    prev = os.environ.get("WHISPERX_DATA_DIR")
    os.environ["WHISPERX_DATA_DIR"] = tmp
    try:
        from app import server  # imported after mocks so the warm thread is harmless
        yield server
    finally:
        if prev is None:
            os.environ.pop("WHISPERX_DATA_DIR", None)
        else:
            os.environ["WHISPERX_DATA_DIR"] = prev


@pytest.fixture
def spies(server_mod, monkeypatch):
    """Replace the two background ops + oauth + periodic start with recorders.

    Leaves ``assess_link`` for each test to set per the outcome under test.
    """
    calls = {"seed": 0, "restore": 0}
    monkeypatch.setattr(server_mod, "_seed_initial_backup",
                        lambda: calls.__setitem__("seed", calls["seed"] + 1))
    monkeypatch.setattr(server_mod, "_seed_initial_restore",
                        lambda: calls.__setitem__("restore", calls["restore"] + 1))
    # The consent flow + periodic loop are out of scope here.
    import app.backup.oauth as oauth
    monkeypatch.setattr(oauth, "link_interactive", lambda *a, **k: None)
    monkeypatch.setattr(server_mod._backup, "is_linked", lambda: True)
    monkeypatch.setattr(server_mod._backup, "start_periodic", lambda: None)
    return calls


def _assessment(outcome, exists):
    return LinkAssessment(outcome, RemoteState(exists=exists, generation=1,
                                               entries=2, total_size=10,
                                               created_at="2026-01-01T00:00:00+00:00"))


# (outcome, remote exists, expect seed?, expect restore?, expect prompt?)
_CASES = [
    (LinkOutcome.FRESH, False, 1, 0, False),
    (LinkOutcome.REMOTE_ONLY, True, 0, 1, False),
    (LinkOutcome.IN_SYNC, True, 0, 0, False),
    (LinkOutcome.DIVERGED, True, 0, 0, True),
]


@pytest.mark.parametrize("outcome,exists,seed,restore,prompt", _CASES)
def test_run_backup_link_dispatch(server_mod, spies, monkeypatch,
                                  outcome, exists, seed, restore, prompt):
    """The SSE flow seeds/restores per outcome and only passes `remote` (-> the
    adopt/overwrite prompt) on a real conflict."""
    rendered = []
    monkeypatch.setattr(server_mod, "_render_backup",
                        lambda template, **ctx: rendered.append(ctx) or "<div>card</div>")
    monkeypatch.setattr(server_mod._backup, "assess_link",
                        lambda: _assessment(outcome, exists))

    server_mod._run_backup_link("partials/_backup_card.html")

    assert spies["seed"] == seed
    assert spies["restore"] == restore
    # Exactly one render; `remote` (the prompt trigger) only on DIVERGED.
    assert len(rendered) == 1
    assert (rendered[0].get("remote") is not None) is prompt
    # A terminal "linked" event is recorded for late SSE subscribers.
    assert server_mod._backup_link["result"]["status"] == "linked"


@pytest.mark.parametrize("outcome,exists,seed,restore,prompt", _CASES)
def test_backup_link_json_dispatch(server_mod, spies, monkeypatch,
                                   outcome, exists, seed, restore, prompt):
    """The JSON endpoint returns the outcome and fires the same auto op; a
    DIVERGED remote is left for the client to resolve via /backup/bootstrap/*."""
    monkeypatch.setattr(server_mod._backup, "assess_link",
                        lambda: _assessment(outcome, exists))

    client = server_mod.app.test_client()
    resp = client.post("/backup/link")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["linked"] is True
    assert body["outcome"] == outcome.value
    assert body["remote"]["exists"] is exists
    assert spies["seed"] == seed
    assert spies["restore"] == restore


# --- live sync status (persistent SSE + async manual push) -------------------

def test_backup_status_event_shape(server_mod):
    """The status payload carries the re-rendered card for the persistent stream."""
    ev = server_mod._backup_status_event()
    assert ev["type"] == "backup"
    assert isinstance(ev["html"], str) and ev["html"]
    assert ev["state"] == server_mod._backup.status()["state"]


def test_publish_backup_pushes_to_status_channel(server_mod):
    """on_change -> _publish_backup -> a delivered event on the status channel."""
    q = server_mod._broker.subscribe(server_mod.BACKUP_STATUS_CHANNEL)
    try:
        server_mod._publish_backup()
        event = q.get(timeout=2)
    finally:
        server_mod._broker.unsubscribe(server_mod.BACKUP_STATUS_CHANNEL, q)
    assert event["type"] == "backup"
    assert "html" in event


def test_status_events_emits_current_state_on_connect(server_mod):
    """GET /backup/status/events sends the current card immediately on connect."""
    client = server_mod.app.test_client()
    resp = client.get("/backup/status/events", buffered=False)
    chunk = next(resp.response)  # first SSE frame, then we stop reading
    resp.close()
    text = chunk.decode() if isinstance(chunk, bytes) else chunk
    assert text.startswith("data: ")
    assert '"type": "backup"' in text


def test_settings_backup_now_is_async(server_mod, monkeypatch):
    """Manual push kicks a background runner and returns the card immediately."""
    called = {"n": 0}
    monkeypatch.setattr(server_mod._backup, "is_linked", lambda: True)
    monkeypatch.setattr(server_mod, "_run_backup_async",
                        lambda: called.__setitem__("n", called["n"] + 1))
    client = server_mod.app.test_client()
    resp = client.post("/settings/backup/now")
    assert resp.status_code == 200
    assert called["n"] == 1


def test_backup_now_json_async_and_guarded(server_mod, monkeypatch):
    """JSON /backup/now: 409 when unlinked, else 202 + kicks the async runner."""
    monkeypatch.setattr(server_mod._backup, "is_linked", lambda: False)
    client = server_mod.app.test_client()
    assert client.post("/backup/now").status_code == 409

    called = {"n": 0}
    monkeypatch.setattr(server_mod._backup, "is_linked", lambda: True)
    monkeypatch.setattr(server_mod, "_run_backup_async",
                        lambda: called.__setitem__("n", called["n"] + 1))
    resp = client.post("/backup/now")
    assert resp.status_code == 202
    assert called["n"] == 1
