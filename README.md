<div align="center">

# WhisperXGUI for Windows  
### Real-time multilingual transcription + smart model selection — built for Microsoft Hackathon 2025 energy and velocity ⚡

</div>

Turn raw audio or full-length videos into timestamped, speaker-aware, multi-format subtitles in minutes. A sleek WPF desktop client + a turbocharged PowerShell runner script wrap the WhisperX pipeline with hardware-aware intelligence, automatic model sizing, GPU fallbacks, and zero-fuss video handling.

## 🚀 Why This Project
We wanted a Windows-first, hackathon-ready tool that:
- Works out-of-the-box on clean dev laptops
- Accepts huge `.mp4` / `.mkv` / `.wav` inputs seamlessly
- Picks a Whisper model intelligently (no memorizing model zoo names)
- Survives flaky CUDA / cuDNN installs with graceful CPU fallback
- Gives fast iteration + reproducible results

## ✨ Highlights
- Auto model selection (VRAM/RAM aware)
- Video → audio extraction (ffmpeg) with optimal 16kHz mono normalization
- Smart compute type fallback (`float16` → `float32` → user override)
- Diarization & VAD with safer Silero fallback
- Single-script command-line UX (`run_whisperx.ps1`)
- GUI powered by .NET 8 (WPF) for demos & non-CLI users
- Resilient to broken cuDNN installs (auto-detect + advise)

## 🏁 Hackathon Project Summary (Microsoft Global Hackathon 2025)
### What We Built
An end-to-end, Windows-first packaging of WhisperX + faster-whisper that delivers fast, accurate, multilingual transcription with word-level timestamps, speaker diarization, and automatic resource-aware model sizing. It ships as BOTH:
- A PowerShell one-liner (`run_whisperx.ps1`) for automation & scripting.
- A lightweight .NET 8 WPF desktop app (`WhisperXGUI`) for non-technical users.

Everything runs locally on a developer or knowledge worker’s PC—no sensitive training or lecture recordings need to leave the device, avoiding confidentiality and compliance risk while still benefiting from cutting-edge AI speech technology.

### Who It’s For
People and teams who maintain libraries of internal trainings, technical briefings, onboarding calls, lecture recordings, or customer education sessions and need fast, private generation of subtitles or multilingual transcripts (e.g. English source → localized subtitle sets) without waiting on cloud job queues or exposing confidential content.

### How It Works (Demo Flow)
1. User selects a video/audio file in the GUI (or passes a path to the script).
2. Video (if present) is auto-converted → normalized 16 kHz mono WAV via ffmpeg.
3. Hardware probe picks the largest safe Whisper model (overrideable) based on VRAM / RAM headroom.
4. Batched faster-whisper inference yields high-throughput token decoding (up to ~70× realtime with `large-v2` under optimal GPU conditions).
5. Optional diarization + VAD pipeline segments speakers and reduces hallucinations.
6. wav2vec2 alignment refines raw utterance timestamps down to accurate word-level timing.
7. Outputs (SRT / VTT / TXT / others) are written to the chosen directory; temporary artifacts cleaned up.

### Impact
| Stakeholder | Value |
|-------------|-------|
| Individual creators | Rapid local subtitle generation for publishing & accessibility |
| Enterprise trainers | Confidential content never leaves managed endpoints |
| Global audiences | Higher comprehension via accurate timing + speaker labels |
| Microsoft ecosystem | Showcases Windows as a performant AI workstation (hybrid CPU/GPU resilience) |
| Sustainability | Local batching reduces redundant multi-upload cloud cycles |

### Key Technical Differentiators
| Area | Differentiator | Why It Matters |
|------|---------------|----------------|
| Model Sizing | Automatic VRAM/RAM probing → picks safest largest model | Removes guesswork; prevents OOMs |
| Throughput | Batched faster-whisper backend | 10-70x realtime depending on GPU |
| Accuracy | wav2vec2 forced alignment for word‑level timestamps | High‑precision subtitle syncing |
| Robustness | Graceful fallback: CUDA → CPU; pyannote → Silero | Keeps results flowing despite env issues |
| Privacy | 100% on‑device | No data egress / compliance friendly |
| UX | Dual interface (CLI + WPF GUI) | Fits both power users & general staff |
| Maintainability | Pinned deps via `uv` | Reproducible environments |

### Glossary (Plain Language)
| Term | Meaning |
|------|---------|
| ASR (Automatic Speech Recognition) | Turning spoken audio into text |
| Word‑level timestamps | Precise start/end time for each word, not just whole sentences |
| Forced alignment | Adjusting rough transcript timings to the actual audio using a secondary model (e.g. wav2vec2) |
| VAD (Voice Activity Detection) | Detecting where speech starts/stops to cut silence & reduce hallucinations |
| Speaker diarization | Labeling which speaker talks when (Speaker 1, Speaker 2, …) |
| Phoneme | Smallest sound unit that distinguishes words (e.g. /p/ in “tap”) |

### Why WhisperX vs Plain Whisper
OpenAI’s base Whisper provides strong transcriptions but only coarse (utterance‑level) timestamps and no native batching. WhisperX enhances this with batching, alignment for accurate per‑word timing, and integrated diarization & VAD—making outputs more directly usable for high‑quality subtitle authoring and analytics.

---

## 🔧 5‑Minute Quick Start
```pwsh
# (Admin) Core prerequisites
winget install -e --id Python.Python.3.11 --scope machine
winget install -e --id astral-sh.uv
winget install -e --id Gyan.FFmpeg
winget install -e --id Microsoft.DotNet.SDK.8
winget install -e --id Nvidia.CUDA   # Optional (GPU)

# Reboot if you added CUDA, then verify (optional)
nvidia-smi

# Clone / enter project root
cd D:\WhisperX_for_Windows

# Install Python deps via uv (editable + pinned)
uv pip install -e .

# Sanity check
python -m whisperx --help

# Launch GUI (dev mode)
dotnet run --project WhisperXGUI/WhisperXGUI.csproj

# Or go CLI with auto model + video handling
./run_whisperx.ps1 .\sample.mp4 --language en --output_format srt
```
Want a specific interpreter? Set once:
```pwsh
setx WHISPERX_PYTHON "C:\\Path\\To\\python.exe"
```

## 🖥 GUI Workflow (WhisperXGUI)
1. Open the app (`dotnet run` or built EXE under `WhisperXGUI\bin\Release`).  
2. Select your media file (`.mp4`, `.mkv`, `.wav`, etc.).  
3. Choose output folder + format(s).  
4. (Optional) Language code (`en`, `zh`, `fr`, ...).  
5. (Optional) Enable diarization.  
6. Device auto-detect shows `CPU` or `CUDA`.  
7. Run → watch logs → collect subtitles.  

## 🛠 CLI Power Script (`run_whisperx.ps1`)
Core features:
- Auto audio extraction from video
- Fast ffmpeg normalization (16kHz mono WAV)
- Temporary file cleanup
- Auto Whisper model selection (see below)
- Automatic `--compute_type` safety on CPU
- Silero VAD fallback when pyannote mismatches detected

### Quick Examples
```pwsh
# Chinese transcription to SRT
./run_whisperx.ps1 meeting.wav --language zh --output_format srt

# Video → auto-extract → VTT + TXT outputs
./run_whisperx.ps1 keynote.mp4 --language en --output_format all

# Speaker diarization (limit speakers)
./run_whisperx.ps1 panel.mkv --language en --diarize --max_speakers 5

# Force quantized CPU inference
./run_whisperx.ps1 audio.wav --compute_type int8 --device cpu
```

### Supported Formats
Audio: `.wav`, `.mp3`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`  
Video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.mpg`, `.mpeg`

### ffmpeg Processing Defaults
| Setting    | Value            | Rationale                       |
|-----------|------------------|---------------------------------|
| SampleRate| 16000 Hz         | Stable STT performance          |
| Channels  | 1 (mono)         | Lower memory + deterministic    |
| Format    | 16‑bit PCM WAV    | Lossless for ASR front-end      |

## 🤖 Automatic Whisper Model Selection
If you do NOT pass `--model` the script chooses one:
1. Environment override: `WHISPERX_DEFAULT_MODEL` wins
2. `--no-auto-model` skips logic
3. Hardware probe → VRAM (GPU) else system RAM + core count
4. Largest safe model selected (headroom minded)

| Model    | Comfort Min (RAM/VRAM) |
|----------|------------------------|
| tiny     | 1 GB                   |
| base     | 2 GB                   |
| small    | 4 GB                   |
| medium   | 8 GB                   |
| large-v2 | 16 GB                  |
| large-v3 | 18 GB                  |

Examples:
```pwsh
./run_whisperx.ps1 clip.wav --language en          # auto size
$env:WHISPERX_DEFAULT_MODEL='medium'
./run_whisperx.ps1 clip.wav --language en          # forced medium via env
./run_whisperx.ps1 clip.wav --no-auto-model        # underlying default
./run_whisperx.ps1 clip.wav --model large-v2       # explicit beats auto
```
Want reproducibility in CI? Set the env var.  

## 🧮 Compute Type Logic
| Scenario | Action |
|----------|--------|
| GPU + user forces `--device cuda` + cuDNN present | Leave default (usually `float16`) |
| GPU present but cuDNN missing | Install/advise, fallback to CPU `float32` |
| CPU only & no `--compute_type` | Auto-add `--compute_type float32` |
| User passes explicit compute type | Respected |

Manual overrides:
```pwsh
./run_whisperx.ps1 sample.wav --compute_type int8 --device cpu
./run_whisperx.ps1 sample.wav --device cuda --compute_type float16
```

## 🧬 Diarization & VAD Strategy
- Default favors reliability: Silero VAD when pyannote model mismatch warnings appear
- Use `--diarize` to enable speaker labeling
- Tune speaker bounds: `--min_speakers 1 --max_speakers 6`

## ⚡ Performance Tuning Tips
| Goal | Tip |
|------|-----|
| Faster CPU runs | Use `--compute_type int8` on medium/small models |
| Reduce memory pressure | Pick `small` or `base` + `int8` |
| Long multi-hour video | Convert externally to WAV then pass to script (skips inline extraction cost) |
| GPU under-utilized | Increase `--batch_size` (watch VRAM) |
| Avoid pyannote overhead | Force `--vad_method silero` |

## 🩺 Troubleshooting

### Common Windows Issues
| Symptom | Meaning | Fix |
|---------|---------|-----|
| `uv: command not found` | winget PATH not updated | Restart PowerShell or run `refreshenv` |
| `ffmpeg not found in PATH` | FFmpeg not installed/accessible | Restart PowerShell after winget install |
| `ModuleNotFoundError: torch` | Wrong Python interpreter | Use `.\.venv\Scripts\python.exe -m whisperx` |
| `Video file detected but ffmpeg not available` | FFmpeg installation issue | `winget install -e --id Gyan.FFmpeg --scope machine` |
| Long download pause | Model downloading (1-5GB) | Be patient, models cache for future use |
| `Could not locate cudnn_ops_infer64_8.dll` | cuDNN not installed | `uv pip install nvidia-cudnn-cu12` |
| Float16 error on CPU | No GPU path | Script auto-adds `float32` – or add manually |
| Pyannote version mismatch spam | Legacy model metadata | Add `--vad_method silero` |
| Permission denied errors | Antivirus blocking | Add project folder to antivirus exclusions |

### Quick Fixes
```powershell
# Refresh PATH environment 
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Force reliable CPU mode
.\run_whisperx.ps1 video.mp4 --device cpu --compute_type float32 --vad_method silero

# Clean reinstall if issues persist
Remove-Item -Recurse -Force .venv
uv venv && uv pip install -e .
```

### Getting Help
1. **Check Setup**: Run `.\setup_whisperx.ps1` to verify installation
2. **Read Guide**: See `WINDOWS_SETUP_GUIDE.md` for detailed troubleshooting  
3. **Test Components**: Verify `python`, `ffmpeg`, `dotnet`, `uv` individually
4. **Use CPU Mode**: Add `--device cpu` for maximum compatibility

Manual cuDNN install:
```pwsh
uv pip install --python .\.venv\Scripts\python.exe nvidia-cudnn-cu12
```

Force CPU for stability:
```pwsh
./run_whisperx.ps1 video.mp4 --device cpu --vad_method silero --compute_type float32
```

## 🗺 Roadmap (Hackathon → Beyond)
- [ ] Optional progress bar per segment
- [ ] Parallel chunk processing for long videos
- [ ] Export JSON w/ word-level timestamps
- [ ] Inline translation mode (`--translate en`)
- [ ] Live microphone capture mode
- [ ] GPU multi-device sharding

## 📚 Documentation

- **`README.md`** - This file with overview and quick start
- **`WINDOWS_SETUP_GUIDE.md`** - Detailed Windows setup and troubleshooting  
- **`AGENTS.md`** - Development guidelines for contributors
- **`setup_whisperx.ps1`** - Automated setup script for Windows
- **`EXAMPLES.md`** - Usage examples and advanced scenarios

## 🤝 Contributing
PRs welcome—keep commits focused; include before/after notes for performance-affecting changes.  
Run formatting + linting before submitting:
```pwsh
black whisperx
ruff check whisperx
```

## 📄 License
This project inherits upstream WhisperX licensing. See `LICENSE` for details.

## 🙌 Acknowledgements
Built for Microsoft Hackathon 2025. Powered by WhisperX, faster-whisper, PyTorch, Silero VAD, and the open-source speech community.

