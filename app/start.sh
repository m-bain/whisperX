#!/usr/bin/env bash
# Run the WhisperX web app natively (no Docker), auto-selecting the best ASR
# backend for the host:
#
#   macOS Apple Silicon (arm64) -> MLX (Apple GPU), via the `mlx` extra
#   Linux x86_64                -> CUDA GPU when available, else CPU
#   anything else               -> CPU
#
# Docker on Mac runs a Linux VM with no Metal passthrough (CPU only), so run
# this on the host to get Apple-GPU acceleration. On a CUDA Linux host this
# gives GPU acceleration too (torch is CUDA-pinned for x86_64 Linux in
# pyproject's [tool.uv.sources]).
#
# Usage:  ./app/start.sh
# Then open http://localhost:5000
#   (on Mac, also set Settings -> Compute Device -> Apple GPU (MLX)).
set -euo pipefail

# Repo root = parent of this script's dir, regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

# --- Pick backend + uv extras per platform ----------------------------------
# Arrays (not strings) so empty == no extra flags. The ${arr[@]+...} guard keeps
# expanding an empty array safe under `set -u` on bash 3.2 (macOS's default).
EXTRA=()
BACKEND="CPU"
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  EXTRA=(--extra mlx)            # installs mlx-whisper (Apple GPU backend)
  BACKEND="Apple GPU (MLX)"
elif [[ "$OS" == "Linux" ]]; then
  BACKEND="CUDA GPU (if present, else CPU)"
fi
echo "Platform: $OS $ARCH  ->  ASR backend: $BACKEND"

# --- ffmpeg (whisperx shells out to it to decode audio; see whisperx/audio.py) ---
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. Install it:" >&2
  case "$OS" in
    Darwin) echo "  brew install ffmpeg" >&2 ;;
    Linux)  echo "  sudo apt-get install -y ffmpeg   (or your distro's package manager)" >&2 ;;
    *)      echo "  install ffmpeg via your platform's package manager" >&2 ;;
  esac
  exit 1
fi

# --- Frontend assets: bundled locally into app/static/vendor (gitignored, no ---
# CDNs). Build with Bun if the bundle is missing. See app/build.ts / app/README.md.
if [[ ! -d app/static/vendor ]]; then
  if ! command -v bun >/dev/null 2>&1; then
    echo "Frontend assets (app/static/vendor) are missing and Bun isn't installed." >&2
    echo "Install Bun from https://bun.sh, then re-run" \
         "(or build once:  cd app && bun install && bun run build)." >&2
    exit 1
  fi
  echo "Bundling frontend assets with Bun…"
  (cd app && bun install && bun run build)
fi

# --- Python deps. Idempotent. On Mac this adds the mlx extra. -----------------
echo "Syncing Python deps…"
uv sync ${EXTRA[@]+"${EXTRA[@]}"}

# Flask + keyring are app-only deps (app/requirements.txt), not part of
# pyproject, so add them ephemerally for this run instead of mutating the
# project env. keyring stores the Hugging Face token in the OS keyring.
PORT="${PORT:-5000}"
echo "Starting WhisperX web app on http://localhost:${PORT} (Ctrl-C to stop)…"
PORT="$PORT" uv run ${EXTRA[@]+"${EXTRA[@]}"} --with "Flask>=3.0" --with "keyring>=24" python -m app.server
