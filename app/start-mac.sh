#!/usr/bin/env bash
# Run the WhisperX web app natively on macOS (Apple Silicon) so the MLX
# (Apple GPU) ASR backend is available. Docker on Mac runs a Linux VM with no
# Metal passthrough, so MLX/CUDA are never usable there — CPU only. Run this
# on the host instead.
#
# Usage:  ./app/start-mac.sh
# Then open http://localhost:5000  (Settings → Compute Device → Apple GPU (MLX)).
set -euo pipefail

# Repo root = parent of this script's dir, regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "MLX needs Apple Silicon macOS (arm64). Detected: $(uname -s) $(uname -m)." >&2
  echo "On other platforms use Docker (CPU) or a CUDA host." >&2
  exit 1
fi

# whisperx shells out to ffmpeg to decode audio (whisperx/audio.py). The Docker
# image apt-installs it; native runs need it on PATH.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH. Install it:  brew install ffmpeg" >&2
  exit 1
fi

# Project env + the 'mlx' extra (installs mlx-whisper). Idempotent.
echo "Syncing deps (with mlx extra)…"
uv sync --extra mlx

# Flask is an app-only dep (app/requirements.txt), not part of pyproject, so add
# it ephemerally for this run instead of mutating the project env.
PORT="${PORT:-5000}"
echo "Starting WhisperX web app on http://localhost:${PORT} (Ctrl-C to stop)…"
PORT="$PORT" uv run --extra mlx --with "Flask>=3.0" python -m app.server
