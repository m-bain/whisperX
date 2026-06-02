# WhisperX web frontend

Minimal Flask + htmx UI for WhisperX: upload audio, run transcription + word
alignment + (optional) speaker diarization on **CPU**, view the speaker-labelled
transcript, and download `srt` / `vtt` / `txt` / `json`.

This is a standalone app that imports `whisperx` as a library. It is **not** part
of the published `whisperx` package.

## Requirements

- `whisperx` installed in the same environment (from the repo root: `uv pip install -e .`).
- `ffmpeg` on `PATH` (whisperx decodes audio via ffmpeg).
- For diarization: an `HF_TOKEN` **and** acceptance of the
  `pyannote/speaker-diarization-community-1` user agreement on Hugging Face.
  Without a token the app still runs (transcription + alignment), diarization off.

## Config

Copy `app/.env.example` to `app/.env` and set `HF_TOKEN` (enables diarization;
leave blank for transcription + alignment only). `app/.env` is gitignored and is
read both by the local server and the Docker container.

```bash
cp app/.env.example app/.env   # then edit HF_TOKEN
```

## Run (local)

```bash
pip install -r app/requirements.txt        # in the whisperx env
python -m app.server                        # serves on http://localhost:5000
```

## Run (Docker)

CPU-only image; `ffmpeg` is baked in. Built from the repo root (`Dockerfile`,
`docker-compose.yml`). Reads `app/.env` via compose `env_file`.

```bash
docker compose up --build                  # http://localhost:5000
```

Downloaded models are cached in the `whisperx-cache` volume, so they are not
re-downloaded on restart. Override the model with `WHISPERX_MODEL` in `app/.env`.

Plain Docker (no compose):

```bash
docker build -t whisperx-web .
docker run --rm -p 5000:5000 --env-file app/.env \
  -v whisperx-cache:/cache whisperx-web
```

The container runs gunicorn with a **single worker** (the job store, model
bundle, and warm-up thread are in-memory process-local state); threads serve
concurrent htmx polls while jobs run one at a time in the app's executor.

Models load in a background thread at startup; the first upload waits until they
are ready. Jobs run **one at a time** (single-worker executor) so concurrent
uploads queue rather than overload the CPU.

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | — | Enables diarization |
| `WHISPERX_MODEL` | `small` | Whisper model size |
| `WHISPERX_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Diarization model |
| `WHISPERX_BATCH_SIZE` | `8` | Transcription batch size |
| `WHISPERX_MAX_UPLOAD_MB` | `200` | Upload size cap |
| `PORT` | `5000` | Server port |

## Layout

- `pipeline.py` — `load_bundle()` (load models once) + `run_job()` (the 3-stage pipeline).
- `jobs.py` — in-memory `JobStore` + single-worker background executor.
- `server.py` — Flask routes; htmx polls `/jobs/<id>/status` until done.
- `render.py` — result dict → speaker-grouped transcript HTML.
- `templates/` — `index.html`, `_status.html` (self-polling), `_result.html`.
