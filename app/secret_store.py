"""Secure, cross-platform storage for the Hugging Face token.

The token is a secret, so it never lands in SQLite or a plaintext file. It is
kept in the OS keyring — macOS Keychain, Windows Credential Locker, or the Linux
Secret Service (GNOME Keyring / KWallet) — via the ``keyring`` library.

There is intentionally **no file fallback**: on a host with no keyring backend
(e.g. a headless Linux box with no Secret Service), :func:`set_hf_token` raises
:class:`SecretStoreUnavailable` with guidance rather than silently writing the
token somewhere readable. Such hosts can still supply the token via the
``HF_TOKEN`` environment variable, which :func:`resolve_hf_token` honours first.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SERVICE = "manuscript-whisperx"
KEY = "hf_token"
# OAuth credentials (JSON) for the cloud backup backend — refresh token + the
# bits google.auth needs to refresh. A secret, so it lives in the keyring too.
GDRIVE_KEY = "google_drive_creds"
# The user's chosen Drive backup folder name. Not a secret, but the keyring is the
# guaranteed non-mirrored, per-machine store already in use — and crucially it must
# NOT live under the data dir, which is itself mirrored to Drive.
GDRIVE_FOLDER_KEY = "google_drive_folder"

# The gated diarization model whose conditions must be accepted for diarization
# to work. Kept here (not imported from pipeline) so this module stays import-cheap.
DIARIZE_MODEL = os.environ.get(
    "WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1"
)

_NO_BACKEND_MSG = (
    "No OS keyring backend is available on this host, so the token can't be "
    "stored securely. Install a Secret Service provider (e.g. gnome-keyring) "
    "or set the HF_TOKEN environment variable instead."
)


class SecretStoreUnavailable(RuntimeError):
    """Raised when no usable OS keyring backend exists."""


def keyring_available() -> bool:
    """Whether a usable (non-fail, non-null) keyring backend is present."""
    try:
        import keyring
        from keyring.backends import fail

        backend = keyring.get_keyring()
        if isinstance(backend, fail.Keyring):
            return False
        # The "null" backend (keyring.backends.null.Keyring) discards writes.
        if backend.__class__.__module__.endswith("backends.null"):
            return False
        return True
    except Exception:  # noqa: BLE001 - any import/probe failure means "unusable"
        return False


def set_hf_token(token: str) -> None:
    """Store the token in the OS keyring. Raises SecretStoreUnavailable if none."""
    token = (token or "").strip()
    if not token:
        raise ValueError("Token is empty.")
    if not keyring_available():
        raise SecretStoreUnavailable(_NO_BACKEND_MSG)
    import keyring

    try:
        keyring.set_password(SERVICE, KEY, token)
    except Exception as exc:  # noqa: BLE001 - surface backend errors uniformly
        raise SecretStoreUnavailable(f"{_NO_BACKEND_MSG} ({exc})") from exc


def get_stored_token() -> str | None:
    """Read the token from the keyring, or None if unset/unavailable."""
    if not keyring_available():
        return None
    import keyring

    try:
        return keyring.get_password(SERVICE, KEY)
    except Exception:  # noqa: BLE001 - treat any read failure as "not stored"
        return None


def delete_hf_token() -> None:
    """Remove the stored token (no-op if absent or no backend)."""
    if not keyring_available():
        return
    import keyring

    try:
        keyring.delete_password(SERVICE, KEY)
    except Exception:  # noqa: BLE001 - PasswordDeleteError when absent, etc.
        pass


def set_gdrive_creds(creds_json: str) -> None:
    """Store the Google Drive OAuth credentials JSON in the OS keyring."""
    creds_json = (creds_json or "").strip()
    if not creds_json:
        raise ValueError("Credentials are empty.")
    if not keyring_available():
        raise SecretStoreUnavailable(_NO_BACKEND_MSG)
    import keyring

    try:
        keyring.set_password(SERVICE, GDRIVE_KEY, creds_json)
    except Exception as exc:  # noqa: BLE001 - surface backend errors uniformly
        raise SecretStoreUnavailable(f"{_NO_BACKEND_MSG} ({exc})") from exc


def get_gdrive_creds() -> str | None:
    """Read the stored Google Drive credentials JSON, or None if unset."""
    if not keyring_available():
        return None
    import keyring

    try:
        return keyring.get_password(SERVICE, GDRIVE_KEY)
    except Exception:  # noqa: BLE001 - treat any read failure as "not stored"
        return None


def delete_gdrive_creds() -> None:
    """Remove stored Google Drive credentials (no-op if absent / no backend)."""
    if not keyring_available():
        return
    import keyring

    try:
        keyring.delete_password(SERVICE, GDRIVE_KEY)
    except Exception:  # noqa: BLE001 - PasswordDeleteError when absent, etc.
        pass


def set_gdrive_folder(name: str) -> None:
    """Store the chosen Drive backup folder name in the keyring."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Folder name is empty.")
    if not keyring_available():
        raise SecretStoreUnavailable(_NO_BACKEND_MSG)
    import keyring

    try:
        keyring.set_password(SERVICE, GDRIVE_FOLDER_KEY, name)
    except Exception as exc:  # noqa: BLE001 - surface backend errors uniformly
        raise SecretStoreUnavailable(f"{_NO_BACKEND_MSG} ({exc})") from exc


def get_gdrive_folder() -> str | None:
    """Read the stored Drive backup folder name, or None if unset."""
    if not keyring_available():
        return None
    import keyring

    try:
        return keyring.get_password(SERVICE, GDRIVE_FOLDER_KEY)
    except Exception:  # noqa: BLE001 - treat any read failure as "not stored"
        return None


def delete_gdrive_folder() -> None:
    """Remove the stored Drive backup folder name (no-op if absent / no backend)."""
    if not keyring_available():
        return
    import keyring

    try:
        keyring.delete_password(SERVICE, GDRIVE_FOLDER_KEY)
    except Exception:  # noqa: BLE001 - PasswordDeleteError when absent, etc.
        pass


def resolve_hf_token() -> str | None:
    """The token in effect: the ``HF_TOKEN``/``HUGGINGFACE_TOKEN`` env var if set
    (it keeps precedence so an operator override always wins), else the keyring."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env:
        return env
    return get_stored_token()


def verify_token(token: str) -> tuple[bool, str]:
    """Live-check a token against Hugging Face.

    Confirms (1) the token is valid (``whoami``) and (2) it can access the gated
    diarization model :data:`DIARIZE_MODEL` (i.e. its user conditions have been
    accepted) — the most common real failure. Returns ``(ok, detail)`` where
    ``detail`` is a user-facing message.
    """
    token = (token or "").strip()
    if not token:
        return False, "Enter a token to continue."

    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    api = HfApi()
    try:
        who = api.whoami(token=token)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return False, "Invalid token. Check that you copied it correctly."
        return False, f"Couldn't verify token: {exc}"
    except Exception as exc:  # noqa: BLE001 - network/other
        return False, f"Couldn't reach Hugging Face: {exc}"

    name = who.get("name") if isinstance(who, dict) else None

    model_url = f"https://huggingface.co/{DIARIZE_MODEL}"
    try:
        api.model_info(DIARIZE_MODEL, token=token)
    except GatedRepoError:
        return False, (
            f"Token is valid, but you haven't accepted the conditions for "
            f"{DIARIZE_MODEL}. Open {model_url} and accept them, then retry."
        )
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            return False, (
                f"Token is valid, but lacks access to {DIARIZE_MODEL}. "
                f"Accept the conditions at {model_url} and retry."
            )
        return False, f"Couldn't check model access: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Couldn't check model access: {exc}"

    who_str = f" as {name}" if name else ""
    return True, f"Token verified{who_str} — diarization is ready."
