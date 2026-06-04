"""Browser end-to-end tests for inline turn editing (the real Flask app + Chromium).

These drive the actual UI the way a user does — double-click a turn, type, Enter —
and assert the observable result, including the network/persistence side effects that
unit tests can't see. They exist mainly as a regression guard for a focusout/keydown
re-entrancy bug where a single save fired TWICE (two POSTs, two deltas), which made
Undo appear to do nothing.

Opt-in (needs Flask + Playwright + a browser, and warms a Whisper model on boot), so
it's skipped in the default `pytest tests/` run. To run it:

    uv run --with Flask --with playwright pytest tests/test_editing_e2e.py -v

with WHISPERX_E2E=1 in the environment and Chromium installed
(`playwright install chromium`).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("WHISPERX_E2E"),
    reason="browser e2e; set WHISPERX_E2E=1 (and install Flask + playwright) to run",
)

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

SID = "e2e_demo"


def _seg(start, end, text, speaker, words):
    return {"start": start, "end": end, "text": text, "speaker": speaker,
            "words": [{"word": w, "start": s, "end": e} for w, s, e in words]}


def _seed(data_dir):
    from app.store import SessionStore
    store = SessionStore(data_dir)
    os.makedirs(store.session_dir(SID), exist_ok=True)
    with open(os.path.join(store.session_dir(SID), "audio.wav"), "wb") as f:
        f.write(b"RIFF0000WAVE")
    segments = [
        _seg(0.0, 1.0, "Hello there.", "SPEAKER_00", [("Hello", 0.0, 0.5), ("there.", 0.5, 1.0)]),
        _seg(1.0, 2.0, "How are you?", "SPEAKER_00", [("How", 1.0, 1.4), ("are", 1.4, 1.7), ("you?", 1.7, 2.0)]),
        _seg(2.0, 3.0, "Fine thanks.", "SPEAKER_01", [("Fine", 2.0, 2.5), ("thanks.", 2.5, 3.0)]),
        _seg(3.0, 4.0, "Good.", "SPEAKER_00", [("Good.", 3.0, 4.0)]),
    ]
    with open(store.result_path(SID), "w") as f:
        json.dump({"segments": segments, "language": "en", "duration": 4.0,
                   "num_segments": 4, "diarized": True}, f)
    store.create(SID, filename="E2E demo", audio_filename="audio.wav", options={}, model="small")
    store.mark_done(SID, language="en", diarized=True, model="small", num_segments=4, duration=4.0)
    return store


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def app_url(tmp_path):
    """Seed a done session, launch the real Flask server against it, yield its URL."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    _seed(data_dir)
    port = _free_port()
    env = {**os.environ, "WHISPERX_DATA_DIR": data_dir, "PORT": str(port), "HF_TOKEN": ""}
    proc = subprocess.Popen([sys.executable, "-m", "app.server"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("server exited early:\n" + proc.stdout.read().decode())
            try:
                with urllib.request.urlopen(f"{base}/sessions/{SID}/view", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not become ready")
        yield base, data_dir
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _turn_texts(page):
    return page.eval_on_selector_all(
        "#tr-body .turn__text", "els => els.map(e => e.innerText.trim())")


def _history_len(data_dir):
    path = os.path.join(data_dir, "sessions", SID, "transcript.edits.json")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return len(json.load(f)["history"])


def _edit_turn0(page, text):
    """Drive the inline editor exactly as a user would, return # of POSTs to /turns."""
    posts = []
    page.on("request", lambda r: posts.append(r.url) if "/turns/" in r.url
            and r.method == "POST" else None)
    page.locator('#tr-body .turn[data-turn="0"] .turn__text').dblclick()
    page.keyboard.press("Control+A")
    page.keyboard.type(text)
    page.keyboard.press("Enter")
    page.wait_for_function(
        "t => document.querySelector('#tr-body .turn[data-turn=\"0\"] .turn__text')"
        ".innerText.includes(t)", arg=text.split()[0])
    page.wait_for_timeout(300)  # let any stray duplicate save land before we count
    return posts


def test_single_edit_fires_one_post_one_delta(app_url):
    """Regression: one Enter-save must produce exactly ONE POST and ONE delta.

    Pre-fix this saw two — focusout re-entered save() during removeAttribute()."""
    base, data_dir = app_url
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        page.goto(f"{base}/sessions/{SID}/view", wait_until="networkidle")
        posts = _edit_turn0(page, "Single edit only.")
        assert _turn_texts(page)[0] == "Single edit only."
        assert len(posts) == 1, f"expected 1 POST, got {len(posts)}: {posts}"
        assert _history_len(data_dir) == 1


def test_undo_reverts_a_single_edit(app_url):
    base, data_dir = app_url
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        page.goto(f"{base}/sessions/{SID}/view", wait_until="networkidle")
        _edit_turn0(page, "Changed line.")
        assert page.locator("#undo-btn").get_attribute("disabled") is None
        page.locator("#undo-btn").click()
        page.wait_for_function(
            "() => document.body.innerText.includes('Hello there')")
        assert _turn_texts(page)[0] == "Hello there. How are you?"
        assert page.locator("#undo-btn").get_attribute("disabled") is not None
        assert _history_len(data_dir) == 0   # fully reverted -> overlay dropped


def test_switching_turns_does_not_clobber_destination(app_url):
    """Open turn 0 to edit, then click turn 1 to switch. Turn 1 must NOT be saved or
    collapsed — switching is not an edit. Pre-fix, the in-flight save of turn 0 swapped
    the body and spuriously saved turn 1, dropping its word-level segments."""
    base, data_dir = app_url
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        posts = []
        page.on("request", lambda r: posts.append(r.url) if "/turns/" in r.url
                and r.method == "POST" else None)
        page.goto(f"{base}/sessions/{SID}/view", wait_until="networkidle")

        def span_count(turn):
            return page.eval_on_selector(
                f'#tr-body .turn[data-turn="{turn}"] .turn__text',
                "e => e.querySelectorAll('.seg').length")

        assert span_count(1) == 2  # "Fine thanks." -> two timed word spans

        # Enter edit on turn 0 (no change), switch to turn 1, then bail out (Esc).
        # Switching may move the editor to turn 1, but nothing was changed, so nothing
        # must be persisted and turn 1's segments must be intact afterwards.
        page.locator('#tr-body .turn[data-turn="0"] .turn__text').dblclick()
        page.wait_for_timeout(150)
        page.locator('#tr-body .turn[data-turn="1"] .turn__text').dblclick()
        page.wait_for_timeout(150)
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)

        assert posts == [], f"switching/closing unchanged turns must not POST: {posts}"
        assert span_count(1) == 2, "turn 1 lost its word segments on switch"
        assert span_count(0) == 5, "turn 0 lost its word segments on switch"
        assert _history_len(data_dir) == 0, "merely switching turns created edits"


def test_adding_a_word_keeps_word_level_spans(app_url):
    """Appending one word to a turn must NOT collapse it to a single untimed span:
    the surviving words keep their timed `.seg` spans (so highlight/seek still work)."""
    base, _ = app_url
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        page.goto(f"{base}/sessions/{SID}/view", wait_until="networkidle")

        def timed_spans(turn):
            return page.eval_on_selector_all(
                f'#tr-body .turn[data-turn="{turn}"] .turn__text .seg[data-start]',
                "els => els.length")

        assert timed_spans(0) == 5  # Hello there. How are you?

        page.locator('#tr-body .turn[data-turn="0"] .turn__text').dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("Hello there. How are you today?")
        page.keyboard.press("Enter")
        page.wait_for_function(
            "() => document.body.innerText.includes('today')")
        page.wait_for_timeout(200)

        # The four unchanged leading words keep their timestamps (>= 4 timed spans);
        # pre-fix this was 0 timed spans (one collapsed segment span aside).
        assert timed_spans(0) >= 4, "adding a word wiped the turn's word-level timing"


def test_escape_cancels_without_saving(app_url):
    """Escape must restore the original text and NOT POST (pre-fix it saved)."""
    base, data_dir = app_url
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        page.goto(f"{base}/sessions/{SID}/view", wait_until="networkidle")
        posts = []
        page.on("request", lambda r: posts.append(r.url) if "/turns/" in r.url
                and r.method == "POST" else None)
        page.locator('#tr-body .turn[data-turn="0"] .turn__text').dblclick()
        page.keyboard.press("Control+A")
        page.keyboard.type("THIS SHOULD NOT PERSIST")
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        assert _turn_texts(page)[0] == "Hello there. How are you?"
        assert posts == [], f"Escape should not POST, but did: {posts}"
        assert _history_len(data_dir) == 0
