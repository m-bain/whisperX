# Mac (Apple Silicon) Performance Findings

Investigation into slow transcription + diarization on Apple Silicon, and
whether the MLX backend is actually being used. **Bottom line: no backend bug.
MLX is correctly selected and is the fastest option on Mac for both stages;
perceived slowness is inherent model cost, not misconfiguration.**

Benchmarked 2026-06-05 on an M-series Mac (torch 2.8, MPS available, `mlx_whisper`
installed). Clip: a 300 s, 2-speaker Russian sample (center cut from a 37 min
file). All runs report the *actually resolved* device and the concrete pipeline
class, not what was requested.

## How device maps to backend

| `--device` | ASR backend | VAD / align / diarize |
|---|---|---|
| `cpu` | `FasterWhisperPipeline` (CTranslate2, CPU) | torch on CPU |
| `cuda` | `FasterWhisperPipeline` (CTranslate2, GPU) | torch on CUDA |
| `mlx` | `MLXWhisperPipeline` (mlx-whisper, Apple GPU) | torch on **MPS** (`_torch_device`) |

CTranslate2 has no Metal backend, so there is no whole-pipeline `mps` device —
`mlx` is the only way to get ASR on the Apple GPU and the torch stages onto MPS.

## Transcription

| Model | Device | Backend class | Infer | RTF | Lang |
|---|---|---|---|---|---|
| small | **whispercpp** | `WhisperCppPipeline` | 15.1s | **0.050** | ru ✅ |
| small | mlx | `MLXWhisperPipeline` | 15.7s | 0.052 | ru ✅ |
| small | cpu | `FasterWhisperPipeline` | 26.7s | 0.089 | ru ✅ |
| large-v3-turbo | **whispercpp** | `WhisperCppPipeline` | **21.0s** | **0.070** | ru ✅ |
| large-v3-turbo | mlx | `MLXWhisperPipeline` | 41.7s | 0.139 | ru ✅ |
| large-v3 | **whispercpp** | `WhisperCppPipeline` | **52.6s** | **0.175** | ru ✅ |
| large-v3 | mlx | `MLXWhisperPipeline` | 82.5s | 0.275 | ru ✅ |
| large-v3 | cpu | `FasterWhisperPipeline` | 217.3s | 0.724 | ru ✅ |
| distil-large-v3 | mlx | `MLXWhisperPipeline` | 35.4s | 0.118 | **en ❌** |
| distil-large-v3 | cpu | `FasterWhisperPipeline` | 69.8s | 0.233 | **en ❌** |

**whisper.cpp (Metal) is now the fastest Mac ASR backend** (added via
`device="whispercpp"`, `pywhispercpp`): **1.57× faster than MLX on large-v3**
(RTF 0.175 vs 0.275) at the same Russian quality, and a tie on `small` (the win
grows with model size). Metal is confirmed engaged (whisper.cpp's `ggml_metal`
backend initializes). `pywhispercpp`'s `Model.__del__` logs a harmless
"Exception ignored" at interpreter shutdown only — no effect on output.

**Best Mac config: `large-v3-turbo` on whisper.cpp — RTF 0.070 (≈14× realtime).**
2.0× faster than MLX turbo and 2.5× faster than whisper.cpp large-v3, while
staying **multilingual** with near-large-v3 quality (turbo is a pruned large-v3).
This is the fast lane `distil-large-v3` could not be — distil is English-only and
mis-transcribed the Russian clip.

- **MLX is genuinely used when selected** — the pipeline class is
  `MLXWhisperPipeline` and the resolved device is `mlx`. No silent CPU fallback.
  (Fallback only happens if `mlx_whisper` is missing; then the sidebar chip shows
  "CPU", because it renders the *resolved* device — the UI cannot misreport it.)
- **MLX is faster than CPU, not slower:** 1.7× on small, **2.6× on large-v3**.
  MLX's serial, no-batch loop only loses to a *CUDA* batched run; on a Mac (no
  CUDA) it beats CPU CTranslate2. MLX is the fastest Mac transcription path.
- **`distil-large-v3` is English-only** — it detected `en` on Russian speech and
  produced English (fast but wrong). Unusable for multilingual content. Use it
  only for known-English files.
- **First run feels slowest** because the model downloads/loads (e.g. CPU
  large-v3 `load=35.6s` was mostly that); warm cache removes it.

**Recommendation:** for multilingual work, **use `device="whispercpp"` with
`large-v3-turbo` (RTF ≈ 0.070, ≈14× realtime)** — fastest accurate Mac path,
multilingual, near-large-v3 quality. Use `large-v3` on whisper.cpp (RTF ≈ 0.175)
when you need maximum accuracy. MLX is the fallback if the `whispercpp` extra
isn't installed. `distil-large-v3` is English-only — avoid for multilingual.

## Diarization (pyannote `speaker-diarization-community-1`)

| Config | Infer | RTF | Speakers | Segments |
|---|---|---|---|---|
| `device=cpu` | 141.9s | 0.473 | 2 | 77 |
| `device=mps`, fallback unset | 22.4s | 0.075 | 2 | 77 |
| `device=mps`, `PYTORCH_ENABLE_MPS_FALLBACK=1` | 22.2s | 0.074 | 2 | 77 |

- **MPS is 6.4× faster than CPU** (22 s vs 142 s, ≈13× realtime), with identical
  output (2 speakers, 77 segments).
- **`PYTORCH_ENABLE_MPS_FALLBACK` makes no difference and is not needed** —
  community-1 runs end-to-end on MPS with no op errors and no CPU fallback.
- The app already routes diarization to MPS on the `--device mlx` path. So under
  `mlx`, diarization is already at RTF ≈ 0.075. The slow case is `--device cpu`
  (the Mac CLI default), where diarization stays on CPU.

**Recommendation:** ensure diarization runs on MPS (use `--device mlx`; in the
web app, select the Apple GPU device). A possible enhancement is to decouple the
diarize device from the ASR device so CPU-ASR users can still diarize on MPS.

## Why the frontend "feels slow"

Not a bug. The web pipeline stacks **transcribe + align + diarize**. With
multilingual large-v3 (transcribe RTF ≈ 0.275) plus align and diarize, a long
file takes minutes per stage. On a 37 min recording that is roughly:
~10 min transcribe + align + ~3 min diarize — large but expected for this model
on Apple Silicon. Levers are model choice (above), not a backend swap.

## Whether to switch backends

- **Diarization:** the originally-planned sherpa-onnx (CoreML) switch is **likely
  not worth the complexity** — MPS already gives ≈13× realtime with zero quality
  change. Revisit only if sub-0.075 RTF is required.
- **Transcription:** **whisper.cpp (Metal) — DONE.** Implemented as
  `device="whispercpp"` (new `whisperx/asr_whispercpp.py`, mirroring the MLX
  backend; CLI `--device whispercpp`; `whispercpp` pyproject extra; app device
  selector). Benchmarked 1.57× faster than MLX on large-v3 at equal quality, so
  it's the recommended default Mac ASR path. lightning-whisper-mlx not pursued.

## Reproducing

The benchmark harnesses live in `/tmp/bench_transcribe.py` and
`/tmp/bench_diarize.py` (not committed). Run from the repo root with
`PYTHONPATH=.`, e.g.:

```bash
PYTHONPATH=. uv run python /tmp/bench_transcribe.py mlx large-v3   # via app ModelManager
PYTHONPATH=. uv run python /tmp/bench_cpp.py whispercpp large-v3   # via whisperx.load_model
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. uv run python /tmp/bench_diarize.py mps
```

The Russian sample clips under `samples/` are **not** committed (large media).
