# CPU-only image for the WhisperX Flask/htmx frontend (app/), built with uv.
#
# torch is pinned to the PyTorch CPU wheel index. We use `uv pip install`
# (not `uv sync`): pyproject's [tool.uv.sources] routes torch to the CUDA
# (cu128) index on x86_64 Linux, but that only applies to uv's project
# resolver (sync/lock). `uv pip` is pip-compatible and ignores those sources,
# so installing CPU torch first keeps the image GPU-free without touching
# pyproject.
# --- Frontend assets: bundle the web app's JS/CSS deps with Bun --------------
# Output (app/static/vendor/) is gitignored, so the image builds it here and the
# final stage copies it in. See app/build.ts and app/package.json.
FROM oven/bun:1 AS assets
WORKDIR /assets
COPY app/package.json app/bun.lock ./
RUN bun install --frozen-lockfile
COPY app/build.ts ./
COPY app/src ./src
RUN bun run build          # -> /assets/static/vendor

# --- Application image -------------------------------------------------------
FROM python:3.13-slim

# uv binary (pinned to match CI: .github/workflows uses 0.11.6).
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# ffmpeg: whisperx shells out to it to decode audio (and torchcodec links its libs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    # Copy instead of hardlink (build cache and target venv are on different layers).
    UV_LINK_MODE=copy \
    # Use the base image's interpreter; don't fetch a managed Python.
    UV_PYTHON_DOWNLOADS=never \
    # Persist downloaded models here (mount a volume to avoid re-downloading).
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    # Durable session storage (audio + transcripts). Mount a volume here.
    WHISPERX_DATA_DIR=/data

WORKDIR /opt/whisperx

# Dedicated venv on PATH so every subsequent `uv pip` / runtime call uses it.
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# 1) CPU torch stack first, from the CPU wheel index (matches pyproject pins).
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch~=2.8.0" "torchaudio~=2.8.0" "torchvision~=0.23.0"

# 2) whisperx itself + frontend deps. torch is already satisfied, so the CPU
#    build is kept. unsafe-best-match lets uv pick newer versions from PyPI even
#    when a package (e.g. tqdm) also appears on the CPU torch index.
COPY pyproject.toml README.md ./
COPY whisperx ./whisperx
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match . \
    && uv pip install "Flask>=3.0" "gunicorn>=22.0"

# 3) The frontend app.
COPY app ./app

# Bundled vendor assets from the `assets` stage (gitignored, so not in COPY app above).
COPY --from=assets /assets/static/vendor ./app/static/vendor

EXPOSE 5000

# Single worker: the JobStore, the model bundle, and the warm-up thread are all
# process-local in-memory state. Threads handle concurrent htmx polls; the heavy
# job runs in app's own single-worker executor. Long startup model load -> generous timeout.
CMD ["uv", "run", "--no-project", "gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "1", "--threads", "8", "--timeout", "120", "app.server:app"]
