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

## Notes

- Diarization and the pyannote VAD/diarization models are gated on Hugging Face — needs `--hf_token` (or `HF_TOKEN`) and accepting the model user agreements.
- `pyproject.toml` carries non-obvious `torchcodec` override-dependencies because it has no wheels for Linux aarch64.
- Version is bumped manually in `pyproject.toml`; releases go through `build-and-release.yml`.
- **Web app (`app/`) — Shoelace + htmx gotcha:** htmx serializes forms via its own value collection, which **skips form-associated custom elements** (`sl-select`, `sl-input`, etc.), so their `name`/`value` silently never reach the server. Don't mirror each control into a hidden input. Instead a single global `htmx:configRequest` listener in `templates/base.html` merges native `FormData(form)` (which *does* include Shoelace controls) into `evt.detail.parameters` (skipping `File` entries — htmx handles those). Any named Shoelace control then "just works"; don't reintroduce per-control hidden mirrors.
- **Web app (`app/`) — htmx binds `hx-*` at process time, not submit time:** a form/element must carry its `hx-post`/`hx-get` etc. **when htmx first processes it** (page load or after `htmx.process()`). Setting `hx-post` later via `setAttribute` does **not** register the request — the element silently falls back to a native submit (page reload / no-op). This bites shared modals whose target URL is per-row (e.g. `/sessions/<id>/rename`): one dialog can't carry a fixed `hx-post`. Fix used in `templates/partials/_rename_modal.html`: drop the htmx form and `fetch()` the per-id endpoint directly from the Save handler, then update the DOM live. (A fixed-URL form like the speaker-rename modal can keep using `hx-post` normally.)
- **Web app (`app/`) — live job progress via Server-Sent Events (not htmx polling):** the upload→transcribe pipeline pushes stage updates (`decoding`/`transcribing`/`loading_align`/`aligning`/`diarizing`) to the browser over SSE. Three pieces: (1) `events.py::Broker` — an in-process, per-`session_id` pub/sub; the background job thread `publish()`es, each open SSE request drains a `subscribe()`d `queue.Queue`. (2) `pipeline.run_job(progress=…)` calls the callback at each stage; `server.py::_on_stage` both `mark_stage`s the SQLite row **and** `broker.publish`es — **the DB row stays the durable source of truth**, the broker only carries live deltas. Terminal (`done`/`error`) events are published by `JobQueue._run` *after* the store is updated, so a client reacting to them reads a consistent row. (3) `GET /sessions/<id>/events` streams `text/event-stream`: it `subscribe()`s, then re-reads the row (closing the subscribe/terminal race), emits current state immediately (so reconnect/late-connect works), relays deltas, sends `:` keepalive comments on idle, and ends on a terminal event. Client side: a global `htmx:load` listener in `base.html` opens one native `EventSource` per `[data-sse-session]` element (idempotent via a `data-sse-init` guard), updates the `[data-status-label]` text from `STAGE_LABELS`, and on a terminal status closes the stream and `htmx.ajax`-fetches the final `/status` render (result panel or error) in place. Don't reach for the htmx SSE extension — it isn't vendored; native `EventSource` + the `htmx:load` hook needs no new dep or `bun` rebuild (the JS lives inline in the `base.html` Jinja template, not the `app/src/` bundle). New columns on the `sessions` table (e.g. `stage`) need a `PRAGMA table_info` + `ALTER TABLE` migration in `SessionStore._migrate` — SQLite has no `ADD COLUMN IF NOT EXISTS`.
