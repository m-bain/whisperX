# Hosting the WhisperX web app online with CUDA

Recommendations for serving the `app/` web frontend (Flask + htmx) online with
GPU acceleration.

## What you're hosting

The `app/` frontend is **stateful and single-process by design**: one gunicorn
worker; the `JobStore`, the loaded model bundle, the warm-up thread, and the SSE
`Broker` are all process-local in-memory state (`Dockerfile:70-74`). Sessions
live in a local SQLite file, uploads on local disk, and jobs run for minutes.

Consequence: **one GPU box, always-on, no horizontal autoscale** unless you
refactor. For personal/demo use that's exactly the right shape.

**GPU footprint:** `large-v2` + wav2vec2 alignment + pyannote diarization load
sequentially, ~8 GB VRAM total. A 12 GB card (RTX 3060 / 4070 / A2000) is a cheap
safe floor; 16 GB+ gives headroom. Requires CUDA 12.8 and **cuDNN 9**
(faster-whisper / CTranslate2 — see `CUDNN_TROUBLESHOOTING.md`).

The current `Dockerfile` is **CPU-only on purpose** (CPU torch wheel index,
`Dockerfile:49-51`). A CUDA image variant is the only real gap to GPU hosting.

## Hosting options

| Option | Fit | Cost shape | Catch |
|--------|-----|-----------|-------|
| **RunPod (community cloud)** | ★ Best. SSH + Docker + NVIDIA drivers + persistent volume + HTTPS proxy. Runs the existing compose almost as-is. | Cheap spot $/hr, always-on. | Community pods can be recycled — persist `/cache`. |
| **Vast.ai** | ★ Best (cheapest). Same shape as RunPod. | Lowest $/hr on the market. | Less reliable hosts; pick high-rated. |
| **Lambda Labs / Hetzner GPU / AWS g5** | Good, more reliable. | Mid, always-on. | Pricier than spot. |
| **Fly.io GPU (a10)** | One-command always-on + managed TLS. | Mid-high. | Less fiddly, costs more. |
| **Modal / Replicate / Baseten** | Scale-to-zero, pay-per-job. | Cheapest if bursty. | Forces a web/GPU **worker split** = refactor. Skip for personal/demo. |
| **K8s + GPU node pool** | Big multi-user scale. | High ops cost. | Overkill for a demo. |

**Recommendation:** a single cheapest GPU pod on **RunPod community** or
**Vast.ai** — deploy the existing container stack unchanged except for a CUDA
image.

## Changes needed to deploy on GPU

1. **`Dockerfile.gpu`** (new) — a copy of `Dockerfile` with two diffs:
   - Base image `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` (the **`-cudnn`**
     variant is mandatory — cuDNN 9 is the #1 GPU load failure) instead of
     `python:3.13-slim`; add Python + ffmpeg + the `uv` binary.
   - Install torch from the **cu128** index
     (`--index-url https://download.pytorch.org/whl/cu128`) instead of the CPU
     index (`Dockerfile:49-51`) — matches the `cu128` pin in `pyproject.toml`
     `[tool.uv.sources]`. Keep everything else, including the **single-worker**
     CMD (`Dockerfile:73-74`) — the in-process state contract requires `--workers 1`.

2. **`docker-compose.gpu.yml`** (new) — a thin override on `docker-compose.yml`
   adding the GPU reservation:
   ```yaml
   services:
     whisperx-web:
       build:
         dockerfile: Dockerfile.gpu
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```
   Run with:
   `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`
   (or `docker run --gpus all` on the pod).

3. **Persist `/cache`** — back the existing `whisperx-cache` volume
   (`docker-compose.yml:13,19`, maps `HF_HOME`) with the pod's **persistent
   network volume** so the multi-GB Whisper / wav2vec2 / pyannote models survive
   spot-pod recycling. Config only.

4. **Lock down access** — the app has **no auth** and will be internet-exposed.
   Put a **Caddy** sidecar (HTTPS + HTTP basic-auth) in front of `:5000`, or use
   the RunPod/Vast HTTPS proxy plus a basic-auth layer. Set
   `WHISPERX_MAX_UPLOAD_MB` (already supported, `app/.env.example`) to cap abuse.
   Put `HF_TOKEN` in `app/.env` to enable diarization.

## Verification

1. **CUDA visible:** in the running container,
   `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
2. **Stage check:** upload a short clip; watch SSE labels go
   `decoding → transcribing → loading_align → aligning → diarizing → done`;
   `nvidia-smi` shows the python process holding VRAM during the job.
3. **Cache persistence:** restart the container — a second job must NOT
   re-download models.
4. **Auth:** the public URL rejects requests without basic-auth credentials.
5. **Diarization:** with `HF_TOKEN` set and the pyannote agreement accepted, a
   multi-speaker file yields labeled speakers.

## Hosting on a Windows machine on a local network

For a Windows box with an NVIDIA GPU serving other devices on your LAN (no
public internet, no cloud). Two paths — Docker is recommended because it reuses
everything above; native is the fallback if you can't run Docker.

**Prerequisites (both paths):**
- A recent **NVIDIA driver** installed on Windows (the driver ships the CUDA +
  cuDNN user-mode libraries WSL/containers use — you do NOT install the CUDA
  toolkit separately for the GPU to work).
- The host's LAN IP: run `ipconfig` and note the IPv4 (e.g. `192.168.1.42`).

### Path A — Docker Desktop + WSL2 (recommended)

1. Install **Docker Desktop** with the **WSL2** backend and enable **GPU
   support** (Docker Desktop ≥ 4.x exposes the NVIDIA GPU through WSL2
   automatically once the Windows NVIDIA driver is present). Verify:
   `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
   must list your GPU.
2. Build/run the **GPU stack from above** unchanged — `Dockerfile.gpu` +
   `docker-compose.gpu.yml`:
   `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`
   The container already binds `0.0.0.0:5000` (`Dockerfile:73-74`) and maps the
   port (`docker-compose.yml:5-6`), so it's reachable on the LAN.
3. Allow the port through **Windows Defender Firewall**: create an inbound rule
   for **TCP 5000** (PowerShell, as admin):
   `New-NetFirewallRule -DisplayName "WhisperX" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow`
4. From any other device on the LAN: open `http://192.168.1.42:5000`.

> On a trusted LAN you can skip the Caddy/basic-auth proxy from the cloud
> section — but anyone on the same network can reach it, so add basic-auth if the
> network isn't fully trusted.

### Path B — Native Windows (no Docker)

Use this if Docker Desktop isn't an option. Note `app/start.sh` is bash
(macOS/Linux only) and **gunicorn doesn't run on Windows** (no `fork`), so the
run command differs.

1. Install **ffmpeg** (`winget install Gyan.FFmpeg` or `choco install ffmpeg`)
   and ensure it's on `PATH` — whisperx shells out to it (`whisperx/audio.py`).
2. Install **uv** (`winget install astral-sh.uv`) and **Bun**
   (for the one-time frontend bundle).
3. From the repo root, install with the CUDA torch stack (matches the `cu128`
   pin in `pyproject.toml`): `uv sync` resolves torch to the cu128 index on
   Windows x86_64 automatically.
4. Build frontend assets once: `cd app && bun install && bun run build`
   (outputs `app/static/vendor`).
5. Run a **Windows-friendly WSGI server** bound to all interfaces (gunicorn
   won't work — use `waitress`):
   `uv run --with Flask --with waitress waitress-serve --listen=0.0.0.0:5000 app.server:app`
   (or, for a quick test only, `set FLASK_RUN_HOST=0.0.0.0 && uv run --with Flask python -m app.server` — Flask's dev server, single-threaded, fine for demo).
   Keep it to **one process** — the in-memory `JobStore`/`Broker` state requires
   it (same single-worker invariant as the container).
6. Put `HF_TOKEN` (for diarization) in `app/.env`.
7. Open the firewall port (step A.3 above) and browse from the LAN to
   `http://<host-ip>:5000`.

**Verify GPU is used:** `nvidia-smi` shows the python process holding VRAM during
a job; or `uv run python -c "import torch; print(torch.cuda.is_available())"`
prints `True`.

## Out of scope (matches personal/demo + deploy-as-is)

Web/GPU-worker split, external queue / object storage, autoscaling,
scale-to-zero, multi-worker gunicorn — all require a refactor and only matter
beyond personal/demo scale.
