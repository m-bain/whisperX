"""Pluggable cloud backup for the WhisperX web app.

Mirrors the local data dir (``sessions.db`` + per-session artifacts) to a
swappable storage backend. See ``service.BackupService`` for the orchestration
and ``backend.StorageBackend`` for the interface.

Public surface:
    BackupService, StorageBackend, RemoteState, BackupResult,
    make_backend_from_env(), build_service(store)

Heavy/optional deps (google libs) are imported lazily inside the gdrive backend,
so importing this package is always cheap.
"""

from __future__ import annotations

import logging
import os

from app import secret_store
from app.backup.backend import StorageBackend
from app.backup.service import (
    BackupResult,
    BackupService,
    BackupState,
    LinkAssessment,
    LinkOutcome,
    RemoteState,
)

logger = logging.getLogger(__name__)

# Default Drive folder when the user hasn't named one. The actual folder is
# user-driven (keyring, not env / not the mirrored data dir) — see gdrive_folder().
DEFAULT_GDRIVE_FOLDER = "Manuscript Backup"

__all__ = [
    "StorageBackend",
    "BackupService",
    "BackupState",
    "BackupResult",
    "RemoteState",
    "LinkAssessment",
    "LinkOutcome",
    "DEFAULT_GDRIVE_FOLDER",
    "gdrive_folder",
    "set_gdrive_folder",
    "make_backend_from_env",
    "build_service",
]


def gdrive_folder() -> str:
    """The Drive backup folder in effect: the user's stored choice, else the default."""
    return secret_store.get_gdrive_folder() or DEFAULT_GDRIVE_FOLDER


def set_gdrive_folder(name: str) -> None:
    """Persist the user's chosen Drive backup folder (blank falls back to default)."""
    secret_store.set_gdrive_folder((name or "").strip() or DEFAULT_GDRIVE_FOLDER)


def make_backend_from_env() -> StorageBackend | None:
    """Construct the configured backend, or None when backup is disabled.

    ``WHISPERX_BACKUP_BACKEND`` selects ``gdrive`` | ``local`` (unset = off).
    """
    kind = (os.environ.get("WHISPERX_BACKUP_BACKEND") or "").strip().lower()
    if not kind:
        return None
    if kind == "local":
        root = os.environ.get("WHISPERX_BACKUP_DIR")
        if not root:
            logger.warning("WHISPERX_BACKUP_BACKEND=local but WHISPERX_BACKUP_DIR "
                           "is unset; backup disabled")
            return None
        from app.backup.local import LocalFsBackend
        return LocalFsBackend(root)
    if kind == "gdrive":
        from app.backup.gdrive import GDriveBackend
        return GDriveBackend(gdrive_folder())
    logger.warning("unknown WHISPERX_BACKUP_BACKEND=%r; backup disabled", kind)
    return None


def build_service(store, *, on_change=None) -> BackupService:
    """Build a BackupService from env config wired to ``store``.

    ``on_change`` is fired on every state transition (see BackupService) so the
    server can broadcast sync status to SSE listeners.
    """
    interval = int(os.environ.get("WHISPERX_BACKUP_INTERVAL", "900"))
    return BackupService(store, make_backend_from_env(), interval=interval,
                         on_change=on_change)
