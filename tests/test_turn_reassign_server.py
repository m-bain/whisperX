"""HTTP surface for turn speaker reassignment + new-speaker enrollment.

Covers the two endpoints added in ``app.server``:

    GET  /sessions/<id>/speakers                     -> picker options
    POST /sessions/<id>/turns/<n>/speaker            -> reassign / enroll

The pure transform (``app.edits.apply_turn_reassign``) and its persistence
(``app.store.save_turn_reassign``) are pinned in test_turn_edits.py; here we
assert the request glue: existing-speaker reassignment, new-speaker enrollment,
and that enrolling a speaker whose name already exists is **rejected** (409).

Like test_backup_server, the server has import-time side effects (a model
warm-up thread, a SessionStore), so we point the data dir at a tmp dir and
neutralize model loading before importing it.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid

import pytest

pytest.importorskip("flask", reason="app dep; see app/requirements.txt")


@pytest.fixture(scope="module")
def server_mod():
    """Import ``app.server`` once, with models + data dir neutralized."""
    from app import pipeline
    pipeline.ModelManager.load_asr = lambda self, name: object()
    pipeline.ModelManager.ensure_diarize = lambda self: None

    tmp = tempfile.mkdtemp(prefix="reassign-server-test-")
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


def _seg(start, end, text, speaker, words):
    return {"start": start, "end": end, "text": text, "speaker": speaker, "words": words}


def _word(token, start, end):
    return {"word": token, "start": start, "end": end}


def _new_session(server_mod):
    """Create a fresh diarized session (A[0,1] B[2] A[3] -> 3 turns) and return its id.

    A unique id per test keeps the module-scoped data dir (and the persistent
    speaker_names table) from leaking state between tests.
    """
    store = server_mod._sessions
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    os.makedirs(store.session_dir(sid), exist_ok=True)
    segments = [
        _seg(0.0, 1.0, "Hello there.", "SPEAKER_00",
             [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)]),
        _seg(1.0, 2.0, "How are you?", "SPEAKER_00",
             [_word("How", 1.0, 1.4), _word("are", 1.4, 1.7), _word("you?", 1.7, 2.0)]),
        _seg(2.0, 3.0, "Fine thanks.", "SPEAKER_01",
             [_word("Fine", 2.0, 2.5), _word("thanks.", 2.5, 3.0)]),
        _seg(3.0, 4.0, "Good.", "SPEAKER_00", [_word("Good.", 3.0, 4.0)]),
    ]
    store.create(sid, "rec.wav", "rec.wav", {}, model="tiny")
    with open(store.result_path(sid), "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f)
    return sid


# --- GET /speakers -----------------------------------------------------------

def test_list_speakers_returns_keys_and_labels(server_mod):
    sid = _new_session(server_mod)
    resp = server_mod.app.test_client().get(f"/sessions/{sid}/speakers")
    assert resp.status_code == 200
    assert resp.get_json() == [
        {"key": "SPEAKER_00", "label": "Speaker 1"},
        {"key": "SPEAKER_01", "label": "Speaker 2"},
    ]


def test_list_speakers_applies_name_overrides(server_mod):
    sid = _new_session(server_mod)
    server_mod._sessions.set_speaker_name(sid, "SPEAKER_00", "Alice")
    data = server_mod.app.test_client().get(f"/sessions/{sid}/speakers").get_json()
    assert {"key": "SPEAKER_00", "label": "Alice"} in data


def test_list_speakers_missing_session_404(server_mod):
    assert server_mod.app.test_client().get("/sessions/nope/speakers").status_code == 404


# --- POST reassign: existing speaker ----------------------------------------

def test_reassign_to_existing_speaker_merges_turns(server_mod):
    sid = _new_session(server_mod)
    client = server_mod.app.test_client()
    # Turn 1 is the lone SPEAKER_01 segment; reassign it to SPEAKER_00.
    resp = client.post(f"/sessions/{sid}/turns/1/speaker", data={"speaker": "SPEAKER_00"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Can-Undo") == "1"
    # Everything is SPEAKER_00 now -> only one distinct speaker remains.
    speakers = client.get(f"/sessions/{sid}/speakers").get_json()
    assert [s["key"] for s in speakers] == ["SPEAKER_00"]


def test_reassign_unknown_turn_400(server_mod):
    sid = _new_session(server_mod)
    resp = server_mod.app.test_client().post(
        f"/sessions/{sid}/turns/99/speaker", data={"speaker": "SPEAKER_00"})
    assert resp.status_code == 400


def test_reassign_missing_speaker_and_name_400(server_mod):
    sid = _new_session(server_mod)
    resp = server_mod.app.test_client().post(f"/sessions/{sid}/turns/0/speaker", data={})
    assert resp.status_code == 400


def test_reassign_missing_session_404(server_mod):
    resp = server_mod.app.test_client().post(
        "/sessions/nope/turns/0/speaker", data={"speaker": "SPEAKER_00"})
    assert resp.status_code == 404


# --- POST reassign: enroll a NEW speaker ------------------------------------

def test_enroll_new_speaker_mints_key_and_assigns(server_mod):
    sid = _new_session(server_mod)
    client = server_mod.app.test_client()
    resp = client.post(f"/sessions/{sid}/turns/1/speaker", data={"name": "Carol"})
    assert resp.status_code == 200
    speakers = client.get(f"/sessions/{sid}/speakers").get_json()
    # A fresh key (one past the highest existing index) labelled "Carol".
    carol = [s for s in speakers if s["label"] == "Carol"]
    assert carol == [{"key": "SPEAKER_02", "label": "Carol"}]


def test_enroll_duplicate_custom_name_rejected(server_mod):
    sid = _new_session(server_mod)
    client = server_mod.app.test_client()
    server_mod._sessions.set_speaker_name(sid, "SPEAKER_00", "Carol")
    # Enrolling another "Carol" would create two indistinguishable speakers.
    resp = client.post(f"/sessions/{sid}/turns/1/speaker", data={"name": "Carol"})
    assert resp.status_code == 409


def test_enroll_duplicate_name_is_case_insensitive(server_mod):
    sid = _new_session(server_mod)
    client = server_mod.app.test_client()
    server_mod._sessions.set_speaker_name(sid, "SPEAKER_00", "Carol")
    resp = client.post(f"/sessions/{sid}/turns/1/speaker", data={"name": "  carol "})
    assert resp.status_code == 409


def test_enroll_duplicate_default_label_rejected(server_mod):
    sid = _new_session(server_mod)
    client = server_mod.app.test_client()
    # "Speaker 1" is SPEAKER_00's default label -> enrolling it is a duplicate.
    resp = client.post(f"/sessions/{sid}/turns/1/speaker", data={"name": "Speaker 1"})
    assert resp.status_code == 409
