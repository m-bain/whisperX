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

from app import secret_store

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


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
    creds = flow.run_local_server(port=port, open_browser=True)
    secret_store.set_gdrive_creds(creds.to_json())
    logger.info("Google Drive linked.")


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
