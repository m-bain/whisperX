"""Locate, version, and (re)vendor the speaker-diarization pipeline.

Diarization runs on the gated ``pyannote/speaker-diarization-community-1``
pipeline. Rather than download it at runtime (which needs a Hugging Face token
and accepted model conditions), a copy is **vendored into the repo** under
:func:`bundled_root` so diarization works out of the box, token-free. pyannote's
``Pipeline.from_pretrained`` loads happily from a local directory, and the
pipeline's ``config.yaml`` references its sub-models with ``$model/…`` (all
local), so the vendored copy is self-contained.

The vendored directory name carries the Hugging Face commit sha as a suffix
(``speaker-diarization-community-1.<sha8>``) so the version is derivable from the
name alone; the full revision + vendor date live in a sibling ``manifest.json``.

A token (when supplied) is only used to **refresh** the model: :func:`vendor`
downloads the latest revision into :func:`data_root` (writable, outside the
package), and :func:`resolve_local_model` prefers that refreshed copy over the
committed baseline. This module is the single source of truth shared by the
vendoring CLI (``app/scripts/vendor_diarize_model.py``) and the Settings
"refresh" endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ID = "pyannote/speaker-diarization-community-1"
# Directory-name prefix for vendored copies; the HF commit sha8 is appended as
# ".<sha8>" so the version is parseable from the name (see derive_version).
MODEL_PREFIX = "speaker-diarization-community-1"
MANIFEST = "manifest.json"


def bundled_root() -> Path:
    """Committed baseline: ``app/models`` (package-relative, CWD-independent)."""
    return Path(__file__).resolve().parent / "models"


def data_root() -> Path:
    """Writable root for refreshed copies: ``<WHISPERX_DATA_DIR>/models``.

    Mirrors ``server.DATA_DIR`` (default ``app/data``, which is gitignored) so a
    refresh never writes into the package dir or dirties the work tree.
    """
    data_dir = os.environ.get("WHISPERX_DATA_DIR", str(Path(__file__).with_name("data")))
    return Path(data_dir) / "models"


def _candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(MODEL_PREFIX + ".") and (p / "config.yaml").is_file()
    ]


def _vendored_at(path: Path) -> str:
    """The manifest vendor date (``YYYY-MM-DD``), or "" if absent/unreadable."""
    try:
        with open(path / MANIFEST) as fh:
            return str(json.load(fh).get("vendored_at", ""))
    except Exception:  # noqa: BLE001 - missing/corrupt manifest sorts oldest
        return ""


def resolve_local_model() -> Optional[Path]:
    """Newest vendored model dir, or None if nothing is vendored.

    A refreshed copy under :func:`data_root` wins over the committed baseline
    under :func:`bundled_root`; within a root, the latest ``vendored_at`` wins.
    """
    # (sort_key, root_priority, path) — data_root (1) beats bundled (0) on ties.
    ranked: list[tuple[str, int, Path]] = []
    for priority, root in ((0, bundled_root()), (1, data_root())):
        for path in _candidates(root):
            ranked.append((_vendored_at(path), priority, path))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked[-1][2]


def derive_version(path: Optional[Path]) -> Optional[dict]:
    """Version info for a vendored dir: sha8 (from the name) + manifest details.

    Returns ``{"sha8", "revision", "vendored_at", "source"}`` or None if ``path``
    is None. ``source`` is "refreshed" when the dir lives under :func:`data_root`,
    else "bundled".
    """
    if path is None:
        return None
    name = path.name
    sha8 = name[len(MODEL_PREFIX) + 1:] if name.startswith(MODEL_PREFIX + ".") else ""
    manifest: dict = {}
    try:
        with open(path / MANIFEST) as fh:
            manifest = json.load(fh)
    except Exception:  # noqa: BLE001
        pass
    try:
        source = "refreshed" if path.resolve().is_relative_to(data_root().resolve()) else "bundled"
    except Exception:  # noqa: BLE001 - is_relative_to needs py3.9+; we target 3.10+
        source = "bundled"
    return {
        "sha8": sha8,
        "revision": manifest.get("revision", sha8),
        "vendored_at": manifest.get("vendored_at", ""),
        "source": source,
    }


def _pyannote_audio_version(snapshot: Path) -> str:
    """Best-effort parse of the pyannote.audio version pinned in config.yaml."""
    try:
        text = (snapshot / "config.yaml").read_text()
        m = re.search(r"pyannote\.audio:\s*(\S+)", text)
        return m.group(1) if m else ""
    except Exception:  # noqa: BLE001
        return ""


def vendor(token: Optional[str], dest_root: Path, revision: Optional[str] = None) -> Path:
    """Download the pipeline from HF and copy it into ``dest_root`` as a vendored dir.

    Resolves ``revision`` from the repo head when not pinned, downloads the
    snapshot (using ``token`` for the gated repo), copies the resolved files
    (following the HF cache's symlinks) into
    ``dest_root/speaker-diarization-community-1.<sha8>``, and writes a manifest.
    Returns the created directory. Idempotent: an existing dir for the same sha
    is replaced.
    """
    from huggingface_hub import HfApi, snapshot_download

    if revision is None:
        revision = HfApi().model_info(REPO_ID, token=token).sha
    snapshot = Path(snapshot_download(REPO_ID, revision=revision, token=token))

    sha8 = revision[:8]
    dest = dest_root / f"{MODEL_PREFIX}.{sha8}"
    dest_root.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    # symlinks=False copies the real blob content the HF cache symlinks point at;
    # ignore HF bookkeeping so only the model files land in the vendored dir.
    shutil.copytree(
        snapshot, dest, symlinks=False,
        ignore=shutil.ignore_patterns(
            ".cache", ".huggingface", "*.lock", ".gitattributes",
            # Not needed to load the pipeline — keep the vendored copy lean.
            "*.md", "*.gif", "*.png", "*.jpg",
        ),
    )

    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "sha8": sha8,
        "vendored_at": date.today().isoformat(),
        "pyannote_audio": _pyannote_audio_version(dest),
    }
    with open(dest / MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    logger.info("Vendored %s @ %s -> %s", REPO_ID, sha8, dest)
    return dest
