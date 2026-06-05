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

from app.backup.backend import StorageBackend
from app.backup.service import BackupResult, BackupService, BackupState, RemoteState

logger = logging.getLogger(__name__)

__all__ = [
    "StorageBackend",
    "BackupService",
    "BackupState",
    "BackupResult",
    "RemoteState",
    "make_backend_from_env",
    "build_service",
]


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
        folder = os.environ.get("WHISPERX_BACKUP_FOLDER", "WhisperX Backup")
        return GDriveBackend(folder)
    logger.warning("unknown WHISPERX_BACKUP_BACKEND=%r; backup disabled", kind)
    return None


def build_service(store) -> BackupService:
    """Build a BackupService from env config wired to ``store``."""
    interval = int(os.environ.get("WHISPERX_BACKUP_INTERVAL", "900"))
    return BackupService(store, make_backend_from_env(), interval=interval)
