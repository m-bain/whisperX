"""Platform-aware location for WhisperX's writable application data.

The web app keeps all durable state — the SQLite session DB, per-session audio
and transcripts, and refreshed diarization models — under one directory. For a
redistributable macOS app this MUST live outside the (read-only, replaced-on-
update) ``.app`` bundle so it survives updates; on Linux/dev we keep the old
package-relative ``app/data`` default for back-compat.

Resolution order:
  1. ``WHISPERX_DATA_DIR`` env var (explicit override; what the packaged macOS
     launcher and the Docker image set).
  2. macOS: ``~/Library/Application Support/WhisperX``.
  3. Otherwise: ``app/data`` next to this package (the historical default).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "WhisperX"


def data_dir() -> Path:
    """The writable root for all durable app state (see module docstring)."""
    override = os.environ.get("WHISPERX_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(__file__).resolve().parent / "data"
