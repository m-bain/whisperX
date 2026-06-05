"""Vendor the gated pyannote diarization pipeline into the repo.

Downloads ``pyannote/speaker-diarization-community-1`` from Hugging Face using a
token and copies it into ``app/models/speaker-diarization-community-1.<sha8>``
(see :mod:`app.diarize_model`). Run once to produce the committed baseline; the
Settings "refresh" flow reuses the same :func:`app.diarize_model.vendor` core to
update into the data dir at runtime.

The model binaries (~32MB) are committed as plain git objects. (Git LFS would be
cleaner, but GitHub forbids LFS uploads on a fork — see the repo's fork status.)

Token resolution order: ``--token`` > ``HF_TOKEN``/``HUGGINGFACE_TOKEN`` env >
``HF_TOKEN`` parsed from ``app/.env``.

Usage:
    uv run --with huggingface_hub python -m app.scripts.vendor_diarize_model
    uv run --with huggingface_hub python -m app.scripts.vendor_diarize_model \
        --dest-root app/data/models --revision <sha>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from app import diarize_model


def _token_from_env_file(env_path: Path) -> str | None:
    """Parse ``HF_TOKEN=…`` from an .env-style file (no external dep)."""
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    except OSError:
        pass
    return None


def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env:
        return env
    # app/.env sits next to the app package (app/.env -> app dir is diarize_model's parent).
    return _token_from_env_file(diarize_model.bundled_root().parent / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", help="HF token (else env / app/.env).")
    parser.add_argument(
        "--dest-root", default=str(diarize_model.bundled_root()),
        help="Where to write the vendored dir (default: app/models, the committed baseline).",
    )
    parser.add_argument("--revision", help="Pin a specific HF commit (default: repo head).")
    args = parser.parse_args(argv)

    token = _resolve_token(args.token)
    if not token:
        print(
            "No Hugging Face token found (--token, HF_TOKEN env, or app/.env). "
            "A token with accepted conditions for\n  "
            f"https://huggingface.co/{diarize_model.REPO_ID}\nis required to download the gated model.",
            file=sys.stderr,
        )
        return 2

    dest = diarize_model.vendor(token, dest_root=Path(args.dest_root), revision=args.revision)
    print(f"Vendored model -> {dest}")
    print(f"Manifest: {dest / diarize_model.MANIFEST}")
    print(f"\nTo update the committed baseline:\n  git add {dest}\n  git commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
