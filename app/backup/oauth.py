"""Google OAuth for the Drive backup backend.

User-consent (installed-app) flow. The OAuth *client* id/secret come from the
environment (you register a Google Cloud OAuth app); the per-user *refresh token*
lands in the OS keyring via :mod:`app.secret_store`, exactly like the HF token —
never in SQLite or a plaintext file.

Scope is ``drive.file`` (least privilege): the app can only see files it created,
so a backup can't read the rest of the user's Drive.

All ``google.*`` imports are lazy so ``import app.backup`` works without the
``gdrive`` extra installed (unit tests use the local backend).
"""

from __future__ import annotations

import logging
import os
import webbrowser
import wsgiref.simple_server
import wsgiref.util
from functools import lru_cache
from pathlib import Path

from app import secret_store

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Styled callback pages served in the browser tab after consent. They live with
# the app's other templates but are read as plain files (the loopback WSGI app
# below isn't Flask, so it can't use render_template).
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


@lru_cache(maxsize=None)
def _callback_html(ok: bool) -> bytes:
    name = "oauth_callback_success.html" if ok else "oauth_callback_error.html"
    return (_TEMPLATE_DIR / name).read_bytes()


def _client_config() -> dict:
    """Installed-app client config from env. Raises if creds aren't provisioned."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set — register a "
            "Google Cloud OAuth app and put them in app/.env."
        )
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def is_linked() -> bool:
    """Whether stored, env-consistent credentials exist."""
    return secret_store.get_gdrive_creds() is not None and _have_client_env()


def _have_client_env() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID")
                and os.environ.get("GOOGLE_CLIENT_SECRET"))


def link_interactive(port: int = 0) -> None:
    """Run the consent flow (opens a browser / loopback) and store credentials.

    Blocking — intended to be triggered by an explicit user action. Persists the
    full authorized-user JSON (refresh token included) to the keyring.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_config(_client_config(), SCOPES)
    creds = _run_consent(flow, port=port, timeout_seconds=300)
    secret_store.set_gdrive_creds(creds.to_json())
    logger.info("Google Drive linked.")


class _CallbackWSGIApp:
    """Loopback redirect handler that serves Manuscript's styled callback page.

    Mirrors ``google_auth_oauthlib``'s internal ``_RedirectWSGIApp`` (it records
    ``last_request_uri`` for the token exchange), but responds with ``text/html``
    so the user sees the designed success page instead of a plain-text line. On a
    denial/error redirect (``?error=…``) it serves the styled error variant.
    """

    def __init__(self) -> None:
        self.last_request_uri: str | None = None

    def __call__(self, environ, start_response):
        self.last_request_uri = wsgiref.util.request_uri(environ)
        ok = "error=" not in (environ.get("QUERY_STRING") or "")
        body = _callback_html(ok)
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]


def _run_consent(flow, *, port: int, timeout_seconds: int):
    """Run the loopback consent flow, serving our styled callback page.

    A reimplementation of ``InstalledAppFlow.run_local_server`` (whose response is
    hardcoded to ``text/plain``) using a ``text/html`` WSGI app. Returns the
    fetched ``google.oauth2.credentials.Credentials``.
    """
    from google_auth_oauthlib.flow import WSGITimeoutError, _WSGIRequestHandler

    host = "localhost"
    wsgi_app = _CallbackWSGIApp()
    wsgiref.simple_server.WSGIServer.allow_reuse_address = False
    local_server = wsgiref.simple_server.make_server(
        host, port, wsgi_app, handler_class=_WSGIRequestHandler
    )
    try:
        flow.redirect_uri = f"http://{host}:{local_server.server_port}/"
        auth_url, _ = flow.authorization_url()
        webbrowser.open(auth_url, new=1, autoraise=True)

        local_server.timeout = timeout_seconds
        local_server.handle_request()

        try:
            # oauthlib insists OAuth 2.0 happen over https.
            authorization_response = wsgi_app.last_request_uri.replace("http", "https")
        except AttributeError as e:
            raise WSGITimeoutError(
                "Timed out waiting for response from authorization server"
            ) from e

        flow.fetch_token(authorization_response=authorization_response)
    finally:
        local_server.server_close()

    return flow.credentials


def load_credentials():
    """Return live, refreshed google credentials, or None if not linked.

    Refreshes an expired access token using the stored refresh token + env client
    secret, and re-persists the rotated credentials.
    """
    raw = secret_store.get_gdrive_creds()
    if not raw or not _have_client_env():
        return None

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_info(_loads(raw), SCOPES)
    # Ensure the client secret from env is applied (stored JSON carries it too,
    # but env is the source of truth for the OAuth app).
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())
        secret_store.set_gdrive_creds(creds.to_json())
    return creds


def unlink() -> None:
    secret_store.delete_gdrive_creds()


def _loads(raw: str) -> dict:
    import json
    return json.loads(raw)
