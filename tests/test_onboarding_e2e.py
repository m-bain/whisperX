"""End-to-end test of the web-app first-run onboarding flow.

Runs the real Flask app (``app.server``) in a background thread and drives it
with Playwright through Welcome -> Access -> Backups -> Engine -> Done, asserting
the gate, the live token verification, and that the choices persist. The Backups
step is optional and (with no backup backend configured in the test) renders a
"not enabled" hint, so the flow just continues past it.

The Hugging Face network is **mocked** at the ``huggingface_hub.HfApi`` layer so
``secret_store.verify_token``'s real branching (valid / invalid / gated) is
exercised without a network call. Heavy Whisper model loading is stubbed out, and
the OS keyring is replaced with an in-memory dict so the test touches nothing
durable.

Skips cleanly if Playwright (or its Chromium build) or Flask isn't installed —
the app deps live in ``app/requirements.txt``, separate from the core project.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time

import pytest

pytest.importorskip("flask", reason="app dep; see app/requirements.txt")
pytest.importorskip("keyring", reason="app dep; see app/requirements.txt")
sync_api = pytest.importorskip("playwright.sync_api", reason="pip install playwright")
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

GOOD = "hf_goodtoken_AAAAAAAAAAAAAAAA"        # valid + pyannote conditions accepted
GATED = "hf_validnotaccepted_BBBBBBBBBBBB"    # valid token, conditions NOT accepted
BAD = "hf_totallyinvalid_CCCCCCCCCCCC"        # not a real token


# --- Mock Hugging Face client (stands in for huggingface_hub.HfApi) ----------
class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def _install_mocks(monkeypatch_env):
    """Patch HF + heavy model loading + keyring before importing the server."""
    from app import pipeline

    # Don't load real Whisper / pyannote models — the warm thread starts at import.
    pipeline.ModelManager.load_asr = lambda self, name: object()
    pipeline.ModelManager.ensure_diarize = lambda self: None

    from app import secret_store
    import huggingface_hub
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    class FakeApi:
        def whoami(self, token=None):
            if token in (GOOD, GATED):
                return {"name": "tester"}
            err = HfHubHTTPError("Invalid user token.")
            err.response = _FakeResponse(401)
            raise err

        def model_info(self, repo_id, token=None):
            if token == GOOD:
                return object()
            # Valid token but the gated model's conditions weren't accepted.
            try:
                raise GatedRepoError("Access to model is restricted.")
            except TypeError:
                err = HfHubHTTPError("Access to model is restricted.")
                err.response = _FakeResponse(403)
                raise err

    huggingface_hub.HfApi = FakeApi  # verify_token does `from huggingface_hub import HfApi`

    # In-memory keyring so we never touch the real OS store.
    mem: dict[str, str] = {}
    secret_store.keyring_available = lambda: True
    secret_store.set_hf_token = lambda t: mem.__setitem__("t", (t or "").strip())
    secret_store.get_stored_token = lambda: mem.get("t")
    secret_store.delete_hf_token = lambda: mem.pop("t", None)
    return mem


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError(f"server didn't come up on port {port}")


@pytest.fixture(scope="module")
def live_server(tmp_path_factory, monkeypatch_module):
    data_dir = str(tmp_path_factory.mktemp("ob-data"))
    monkeypatch_module.setenv("WHISPERX_DATA_DIR", data_dir)
    monkeypatch_module.setenv("WHISPERX_MODEL", "tiny")
    # Neutralize any HF_TOKEN inherited from app/.env so resolve_hf_token() falls
    # through to our in-memory keyring (empty string is falsy but blocks .env load).
    monkeypatch_module.setenv("HF_TOKEN", "")

    mem = _install_mocks(monkeypatch_module)

    from app import server  # imported after mocks so the warm thread is harmless

    port = _free_port()
    thread = threading.Thread(
        target=lambda: server.app.run(host="127.0.0.1", port=port,
                                      threaded=True, use_reloader=False),
        daemon=True,
    )
    thread.start()
    _wait_for_port(port)
    yield {"base": f"http://127.0.0.1:{port}", "server": server, "mem": mem}


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped MonkeyPatch (the built-in one is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def browser():
    try:
        with sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except PlaywrightError as exc:
                pytest.skip(f"Chromium unavailable (run `playwright install chromium`): {exc}")
            yield b
            b.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright runtime unavailable: {exc}")


def _set_token(page, value: str) -> None:
    """Set the Shoelace <sl-input> value and fire sl-input (as a user would)."""
    page.evaluate(
        """(v) => {
            const el = document.getElementById('ob-token');
            el.value = v;
            el.dispatchEvent(new Event('sl-input'));
        }""",
        value,
    )


def test_onboarding_full_flow(live_server, browser):
    base = live_server["base"]
    server = live_server["server"]
    mem = live_server["mem"]
    page = browser.new_page(viewport={"width": 1440, "height": 960})

    # 1) First-run gate: '/' redirects to onboarding.
    page.goto(base + "/")
    assert page.url.rstrip("/").endswith("/onboarding")

    # 2) Welcome -> Access.
    page.click("[data-go='1']")
    page.wait_for_selector("[data-panel='1'].is-active")

    # 3) Invalid token: verification fails, stays on Access.
    _set_token(page, BAD)
    page.click("#ob-verify-btn")
    page.wait_for_selector("#ob-verify .frag--err")
    assert page.locator("[data-panel='1']").get_attribute("class").find("is-active") != -1

    # 4) Valid token but gated-model conditions not accepted: still blocked,
    #    with the actionable message.
    _set_token(page, GATED)
    page.click("#ob-verify-btn")
    page.wait_for_selector("#ob-verify .frag--err")
    assert "conditions" in page.locator("#ob-verify").inner_text().lower()
    assert page.locator("[data-panel='2']").get_attribute("class").find("is-active") == -1

    # 5) Good token: verification passes and advances to Backups (step 3).
    _set_token(page, GOOD)
    page.click("#ob-verify-btn")
    page.wait_for_selector("[data-panel='2'].is-active", timeout=5000)

    # 6) Backups step is optional (no backend configured here) — continue to Engine.
    page.click("[data-panel='2'].is-active [data-go='3']")
    page.wait_for_selector("[data-panel='3'].is-active")

    # 7) Pick a model size + CPU backend, finish to the Done summary (step 5).
    page.click("[data-size='small']")
    page.click("[data-backend='cpu']")
    page.click("[data-go='4']")
    page.wait_for_selector("[data-panel='4'].is-active")
    summary = page.locator(".ob__summary").inner_text()
    assert "Small" in summary and "CPU" in summary
    assert GOOD[-4:] in summary  # masked token shows last 4 chars

    # 7) Enter Manuscript: native form submit -> persist -> redirect to dashboard.
    page.click("#ob-enter")
    page.wait_for_url(base + "/", timeout=5000)
    assert "/onboarding" not in page.url

    # 8) Choices persisted; token stored in (mocked) keyring, not the DB.
    assert server._sessions.get_setting("onboarded") == "1"
    assert server._sessions.get_setting("active_model") == "small"
    assert server._sessions.get_setting("device") == "cpu"
    assert mem.get("t") == GOOD

    page.close()


def test_onboarding_gate_lifts_after_completion(live_server, browser):
    """Once onboarded, '/' serves the dashboard instead of redirecting."""
    base = live_server["base"]
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    page.goto(base + "/")
    # Depends on the full-flow test having completed onboarding (same module scope).
    assert page.url.rstrip("/") == base.rstrip("/")
    assert "Recent Recordings" in page.content()
    page.close()


def test_onboarding_completes_without_token(live_server, browser):
    """The token is optional: leaving it blank still completes onboarding, and
    nothing is written to the keyring — diarization falls back to the bundled model."""
    base = live_server["base"]
    server = live_server["server"]
    mem = live_server["mem"]

    # Reset to a fresh first-run state (module-scoped server shares one DB).
    server._sessions.set_setting("onboarded", "0")
    mem.clear()

    page = browser.new_page(viewport={"width": 1440, "height": 960})
    page.goto(base + "/")
    assert page.url.rstrip("/").endswith("/onboarding")

    # Welcome -> Access, leave the token blank, Continue advances to Backups.
    page.click("[data-go='1']")
    page.wait_for_selector("[data-panel='1'].is-active")
    page.click("#ob-verify-btn")  # blank token -> skip verification
    page.wait_for_selector("[data-panel='2'].is-active", timeout=5000)

    # Backups (optional) -> Engine.
    page.click("[data-panel='2'].is-active [data-go='3']")
    page.wait_for_selector("[data-panel='3'].is-active")

    # Pick a model + backend, finish, enter.
    page.click("[data-size='base']")
    page.click("[data-backend='cpu']")
    page.click("[data-go='4']")
    page.wait_for_selector("[data-panel='4'].is-active")
    assert "Bundled model" in page.locator(".ob__summary").inner_text()

    page.click("#ob-enter")
    page.wait_for_url(base + "/", timeout=5000)

    # Onboarded, choices persisted, and NO token in the keyring.
    assert server._sessions.get_setting("onboarded") == "1"
    assert server._sessions.get_setting("active_model") == "base"
    assert mem.get("t") is None
    page.close()
