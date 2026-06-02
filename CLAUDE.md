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
