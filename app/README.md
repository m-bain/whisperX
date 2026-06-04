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

One command (any platform — bundles the frontend, syncs deps, picks the best ASR
backend: Apple GPU/MLX on Apple Silicon, CUDA on a GPU Linux host, else CPU):

```bash
./app/start.sh                              # serves on http://localhost:5000
```

Or do the steps manually:

```bash
pip install -r app/requirements.txt        # in the whisperx env
(cd app && bun install && bun run build)   # bundle frontend deps -> app/static/vendor/
python -m app.server                        # serves on http://localhost:5000
```

The frontend deps (Shoelace, htmx, the Literata/JetBrains Mono fonts) are
**bundled locally** — no CDNs — by [Bun](https://bun.sh) into `app/static/vendor/`
(gitignored). Run `bun run build` once (and after changing `app/src/` or the
pinned versions in `app/package.json`). The Docker image builds this
automatically, so the step above is only for local runs.

## Run (Docker)

CPU-only image; `ffmpeg` is baked in. Built from the repo root (`Dockerfile`,
`docker-compose.yml`). Reads `app/.env` via compose `env_file`.

```bash
docker compose up --build                  # http://localhost:5000
```

Downloaded models are cached in the `whisperx-cache` volume, so they are not
re-downloaded on restart. `WHISPERX_MODEL` only seeds the **initial** default
model — clients pick a model per upload and switch the global default at runtime
(see *Model selection* below).

Plain Docker (no compose):

```bash
docker build -t whisperx-web .
docker run --rm -p 5000:5000 --env-file app/.env \
  -v whisperx-cache:/cache whisperx-web
```

The container runs gunicorn with a **single worker** (the job store, model
bundle, warm-up thread, and the SSE broker are in-memory process-local state);
threads serve concurrent SSE streams while jobs run one at a time in the app's
executor.

Models load in a background thread at startup; the first upload waits until they
are ready. Jobs run **one at a time** (single-worker executor) so concurrent
uploads queue rather than overload the CPU.

## Live progress (SSE)

The browser watches a job over **Server-Sent Events**, not polling. As the
pipeline advances it pushes the current stage (`decoding` → `transcribing` →
`loading_align` → `aligning` → `diarizing`) to any connected client:

- `events.py::Broker` — in-process, per-`session_id` pub/sub. The background job
  thread `publish()`es; each open SSE request drains a `subscribe()`d queue.
- `run_job(progress=…)` reports each stage; `server.py::_on_stage` writes it to
  the session row **and** publishes it — the SQLite row stays the durable source
  of truth (used for the initial state on connect and after a reload), the broker
  only carries live deltas. Terminal `done`/`error` events are published by
  `JobQueue` *after* the store is updated.
- `GET /sessions/<id>/events` streams `text/event-stream`: current state on
  connect, then deltas, `:` keepalive comments while idle, closing on a terminal
  event.
- Client: a global `htmx:load` hook in `base.html` opens one native `EventSource`
  per `[data-sse-session]` element, updates its label as stages arrive, and on a
  terminal status fetches the final `/sessions/<id>/status` render in place.

## Model selection

A `ModelManager` caches **multiple** Whisper checkpoints at once. Selecting a new
model loads it lazily and keeps previously-used ones in memory, so switching back
is instant (RAM grows with the number of cached models). Diarization and the
wav2vec2 alignment models are loaded once and shared across all Whisper models.

Allowed models (a whitelist — the client cannot trigger arbitrary HF downloads):
`tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`,
`large-v2`, `large-v3`, `distil-large-v3`.

- **Per upload:** the New Recording dialog has a **Model** dropdown; each session
  records the model it used.
- **Global default:** the **Settings** page switches the active model (used when an
  upload doesn't override it). The choice is persisted in SQLite and restored on
  restart.

HTTP API:

| Method | Path | Body | Result |
|--------|------|------|--------|
| `GET`  | `/models` | — | `{active, diarize, models:[{name, loaded, loading, error}]}` |
| `POST` | `/models/active` | `model=<name>` | Switch + persist the active model (warms it); `400` on unknown model |
| `POST` | `/sessions` | `audio`, optional `model=<name>` | Per-upload model override; `400` on unknown model |

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | — | Enables diarization |
| `WHISPERX_MODEL` | `small` | Initial default model (clients switch at runtime; the switch is persisted and takes precedence after first use) |
| `WHISPERX_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Diarization model |
| `WHISPERX_BATCH_SIZE` | `8` | Transcription batch size |
| `WHISPERX_MAX_UPLOAD_MB` | `200` | Upload size cap |
| `PORT` | `5000` | Server port |

## Layout

- `pipeline.py` — `WhisperModel` enum + `ModelManager` (multi-model cache, shared
  diarizer/align) + `run_job(progress=…)` (the 3-stage pipeline, reporting stages).
- `store.py` — `SessionStore`: SQLite metadata (incl. live `stage`) + on-disk
  audio/artifacts, plus a `settings` key/value table (e.g. the persisted active
  model).
- `events.py` — `Broker`: in-process per-session pub/sub bridging the job thread
  to SSE clients.
- `jobs.py` — `JobQueue`: single-worker background executor over the store; emits
  terminal status to the broker.
- `server.py` — Flask routes; `GET /sessions/<id>/events` streams live progress
  over SSE (see *Live progress*); model endpoints `GET /models`,
  `POST /models/active`.
- `render.py` — result dict → speaker-grouped transcript HTML.
- `templates/` — `index.html`, `_status.html` (SSE-driven status), `_result.html`,
  `_models.html` (active-model switcher), `partials/_model_select.html`.
