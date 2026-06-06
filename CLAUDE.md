# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WhisperX is a CLI + Python library for fast, time-accurate speech recognition. It wraps OpenAI Whisper (via the faster-whisper / CTranslate2 backend) and adds three things Whisper lacks: VAD-based batched inference, word-level timestamps via forced phoneme alignment (wav2vec2), and speaker diarization (pyannote-audio).

## Commands

Uses `uv` for dependency management (not plain pip for dev work).

```bash
uv sync --all-extras                 # install project + dev deps (pytest)
uv run pytest tests/ -v              # run all tests
uv run pytest tests/test_word_timestamp_interpolation.py -v   # single test file
uv run pytest tests/test_word_timestamp_interpolation.py::test_name   # single test
uv lock --check                      # verify lockfile is up to date (CI gate)
uv run whisperx audio.mp3 --model large-v2 --diarize --highlight_words True   # run CLI
```

CI (`.github/workflows/`) tests against Python 3.10–3.13. `python-compatibility.yml` enforces `uv lock --check` and a bare `import whisperx` smoke test — keep both green.

### GPU / CUDA

Requires CUDA toolkit 12.8 for GPU. torch is pinned to a CUDA 12.8 index (`cu128`) on x86_64 Linux and CPU index elsewhere — see `[tool.uv.sources]` in `pyproject.toml`. cuDNN library load failures are common; see `CUDNN_TROUBLESHOOTING.md`.

## Architecture

The pipeline runs in three sequential, independently-loaded stages. Each stage loads its own model and the result of one feeds the next:

1. **Transcribe** (`asr.py`) — `load_model()` returns a `FasterWhisperPipeline` (subclass of HF `Pipeline`) wrapping a custom `WhisperModel` (subclass of `faster_whisper.WhisperModel`). VAD (`vads/`) segments audio first, then batches the voiced segments through Whisper. Default ASR options live in `default_asr_options` inside `load_model()`.
2. **Align** (`alignment.py`) — `load_align_model()` + `align()` force-align the transcript to audio with a per-language wav2vec2 model to get word-level timestamps. Language→model mapping is in `DEFAULT_ALIGN_MODELS_TORCH` / `DEFAULT_ALIGN_MODELS_HF`. Alignment is skipped for `--task translate`.
3. **Diarize** (`diarize.py`) — `DiarizationPipeline` (pyannote) labels speakers; `assign_word_speakers()` maps speaker turns onto aligned words.

`transcribe.py::transcribe_task()` orchestrates all three from CLI args. `__main__.py::cli()` defines every CLI flag and is the `whisperx` entry point. The package `__init__.py` exposes the public API (`load_model`, `load_align_model`, `align`, `load_audio`, `assign_word_speakers`) via **lazy imports** — heavy deps (torch, pyannote) are only imported when the function is actually called, so adding a top-level eager import there will slow `import whisperx` and can break the compatibility smoke test.

### Key modules

- `audio.py` — audio loading + mel spectrogram. Defines core constants (`SAMPLE_RATE = 16000`, `N_SAMPLES`, frame/token rates) reused across stages.
- `vads/` — VAD backends behind a common interface; `--vad_method` selects `pyannote` (default) or `silero`.
- `schema.py` — `TypedDict` result shapes (`TranscriptionResult`, `AlignedTranscriptionResult`, `SingleSegment`, `SingleWordSegment`, …). The contract between stages; update here when changing result structure.
- `utils.py` — `get_writer()` and output writers (srt, vtt, txt, tsv, json, aud); also `LANGUAGES` / `TO_LANGUAGE_CODE`.
- `SubtitlesProcessor.py` — subtitle line splitting/formatting.
- `conjunctions.py` — per-language conjunction lists used for sentence/segment splitting.
- `log_utils.py` — `setup_logging()` / `get_logger()`; modules get loggers via `get_logger(__name__)`.

## Conventions

- **Commits:** never add a `Co-Authored-By:` trailer (or any AI co-author line) to commit messages.

## Notes

- Diarization and the pyannote VAD/diarization models are gated on Hugging Face — needs `--hf_token` (or `HF_TOKEN`) and accepting the model user agreements.
- `pyproject.toml` carries non-obvious `torchcodec` override-dependencies because it has no wheels for Linux aarch64.
- Version is bumped manually in `pyproject.toml`; releases go through `build-and-release.yml`.
- **Web app (`app/`) — Shoelace + htmx gotcha:** htmx serializes forms via its own value collection, which **skips form-associated custom elements** (`sl-select`, `sl-input`, etc.), so their `name`/`value` silently never reach the server. Don't mirror each control into a hidden input. Instead a single global `htmx:configRequest` listener in `templates/base.html` merges native `FormData(form)` (which *does* include Shoelace controls) into `evt.detail.parameters` (skipping `File` entries — htmx handles those). Any named Shoelace control then "just works"; don't reintroduce per-control hidden mirrors.
- **Web app (`app/`) — htmx binds `hx-*` at process time, not submit time:** a form/element must carry its `hx-post`/`hx-get` etc. **when htmx first processes it** (page load or after `htmx.process()`). Setting `hx-post` later via `setAttribute` does **not** register the request — the element silently falls back to a native submit (page reload / no-op). This bites shared modals whose target URL is per-row (e.g. `/sessions/<id>/rename`): one dialog can't carry a fixed `hx-post`. Fix used in `templates/partials/_rename_modal.html`: drop the htmx form and `fetch()` the per-id endpoint directly from the Save handler, then update the DOM live. (A fixed-URL form like the speaker-rename modal can keep using `hx-post` normally.)
- **Web app (`app/`) — live job progress via Server-Sent Events (not htmx polling):** the upload→transcribe pipeline pushes stage updates (`decoding`/`transcribing`/`loading_align`/`aligning`/`diarizing`) to the browser over SSE. Three pieces: (1) `sse.py::Broker` — an in-process, per-`session_id` pub/sub; the background job thread `publish()`es, each open SSE request drains a `subscribe()`d `queue.Queue`. (2) `pipeline.run_job(progress=…)` calls the callback at each stage; `server.py::_on_stage` both `mark_stage`s the SQLite row **and** `broker.publish`es — **the DB row stays the durable source of truth**, the broker only carries live deltas. Terminal (`done`/`error`) events are published by `JobQueue._run` *after* the store is updated, so a client reacting to them reads a consistent row. (3) every SSE endpoint streams `text/event-stream` via the shared `sse.py::sse_response(broker, channel, *, initial=, terminal=, pending=)` helper — it owns the `subscribe`/`finally: unsubscribe`, the 15s `:` keepalive comment, and the no-buffering headers, so a new stream only supplies what varies. `initial()` runs *after* subscribe (closing the subscribe/terminal race) and emits current state so reconnect/late-connect works; `terminal(event)` (omit for persistent streams) closes after the last event; `pending()` replays a late-subscriber's already-finished result. Four channels use it: `/sessions/<id>/events` (per-session, terminal `done`/`error`), `/models/events` + `/backup/status/events` (persistent), `/backup/events` (one-shot OAuth, `pending` replay). **When adding a stream, route it through `sse_response` — don't hand-roll the `stream()` loop.** Client side: the primitives live in `static/sse.js` (a plain classic `<script>`, no `bun` rebuild) shared by `base.html` *and* the standalone `onboarding.html`: `openSSE(url, onData)` (EventSource + JSON-parse guard + auto-reconnect), `sseSwap(url, {target, swap, terminal})` (swap a rendered `d.html` fragment into a target + `htmx.process`), and `watchBackupConnect` (built on `sseSwap`). The job-status consumer is a `htmx:load` listener in `base.html` that `openSSE`s one stream per `[data-sse-session]` element (idempotent via a `data-sse-init` guard), updates `[data-status-label]` from `STAGE_LABELS`, and on a terminal status closes + `htmx.ajax`-fetches the final `/status` render in place. **New client streams: reuse `openSSE`/`sseSwap` from `static/sse.js` rather than `new EventSource` inline.** Don't reach for the htmx SSE extension — it isn't vendored; native `EventSource` needs no new dep. **Tests:** the broker has Python unit tests in `tests/test_sse.py` (`uv run pytest`); the `static/sse.js` primitives have frontend unit tests in `app/tests/sse.test.ts` run via `cd app && bun test` (happy-dom + a fake `EventSource`; preload wired in `app/bunfig.toml`, dev dep `@happy-dom/global-registrator`). `static/sse.js` is a **classic script that defines globals** (no `export`s — it's loaded via `<script src>`, not as a module), so the test harness (`app/tests/load-sse.ts`) pulls the functions out by evaluating the source with `new Function`; **don't add `export`/`import` to `sse.js`** or the classic `<script>` load breaks. New columns on the `sessions` table (e.g. `stage`) need a `PRAGMA table_info` + `ALTER TABLE` migration in `SessionStore._migrate` — SQLite has no `ADD COLUMN IF NOT EXISTS`.
- **Tauri boot splash (`packaging/macos/tauri/dist/index.html`):** the Rust shell (`src/main.rs`) creates the window on this **local** page, then `navigate()`s it to `http://127.0.0.1:<port>/` once `/healthz` returns 200. It loads *before* the Flask server is up, so it must be self-contained — **no React/Babel, no `app/static` assets, no localhost deps**. Fonts come from Google Fonts `<link>` (degrade to Georgia/system-mono offline; the real UI self-hosts them via `vendor.css`). The current design is "Loading Screen" option A · Parchment from the Claude Design handoff (`SplashParchment`): a faux borderless light card floating on a parchment backdrop, baked monogram squircle SVG (superellipse `n≈5`, path precomputed — no JS), Literata wordmark + JetBrains Mono tags + shimmer bar. It's a *mock* of a borderless window rendered inside the real (decorated) Tauri window — the stage backdrop fills the actual webview; don't expect `main.rs` to be borderless.
- **Verifying UI with `playwright-cli`:** first **load the skill** (`Skill` tool, `playwright-cli`) before any CLI call — it surfaces the command set and is required for the CLI to work in-session. Then two gotchas: (1) `playwright-cli open` **defaults to `--browser=chrome`, which isn't installed** on this machine → `open --browser=chromium` (the bundled Playwright Chromium). (2) The **`file:` protocol is blocked** — serve static files over http first (`cd <dir> && python3 -m http.server <port> &`) and `goto http://127.0.0.1:<port>/…`. Typical flow: `open --browser=chromium` → `resize W H` → `goto <url>` → `screenshot --filename=…`, then `close` + kill the http server.
