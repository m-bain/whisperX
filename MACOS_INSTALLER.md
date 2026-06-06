# Packaging the WhisperX web app as a macOS installer — research

This document is design research for shipping the web frontend (`app/`) as a
**redistributable macOS app a non-technical user can double-click**, and for
**updating that app without destroying user state** (the SQLite session DB,
per-session audio/transcripts, and the stored Hugging Face token).

It records the constraints, the packaging options with pros/cons, the
recommended design, and the supporting code changes that have already landed.
The macOS-specific build steps (Tauri shell, code signing, notarization, DMG)
must be executed on a Mac with an Apple Developer ID — they are documented here
but not yet implemented.

## Requirements

| Decision | Choice |
|---|---|
| Audience / launch | **Non-technical, double-click `.app`** (drag-to-Applications DMG). No terminal, Homebrew, or pre-installed Python. |
| Signing | **Signed + notarized** with an Apple Developer ID. Gatekeeper must open cleanly. |
| Architecture | **Apple Silicon (arm64) only.** May use MLX / whisper.cpp Metal backends. |
| Bundling | **Lean**: tiny shipped artifact; **Python deps, ffmpeg, and ML models all install on first run.** |
| Window | **Native desktop window** (Tauri / WKWebView), not "just a browser tab". |
| Updates | **Manual re-download of a new DMG.** No Sparkle/auto-update for v1. |

### State that MUST survive an update
- **SQLite DB + session files** — `sessions.db` (WAL mode) and `sessions/<id>/` (audio, transcripts).
- **Refreshed diarization models** — under `<data_dir>/models`.
- **HF token** — macOS Keychain via `keyring`, service `manuscript-whisperx`, key `hf_token`.

The token survives automatically **iff** every release keeps the **same Team ID
and bundle identifier** (Keychain ACLs are bound to the signing identity). The DB
and files survive **iff** they live *outside* the `.app` bundle — hence the data
dir is now relocated (see code changes below).

## Options considered

torch (~2.5–3.5 GB on disk, arm64 CPU/MPS) is an **in-process import** — it can't
be deferred the way ffmpeg and model weights can. So the real choice is *where the
torch stack lives*, which decides the code-signing surface and the update size.

### (A) Frozen `.app` — PyInstaller / py2app — **rejected**
- **Pros**: one self-contained artifact; well-known signing recipe.
- **Cons**: whisperx's lazy/dynamic imports + data files (pyannote/nltk/VAD assets)
  force brittle `hiddenimports`/`collect-all` lists; PyInstaller rewrites dylib load
  paths, **invalidating the ad-hoc signatures inside the torch/ctranslate2 wheels**,
  so hundreds of nested dylibs must be re-signed; ~3 GB bundle; slow notarization.
  Not lean.

### (B) Embedded relocatable interpreter inside the signed `.app` — **fallback**
Ship `python-build-standalone` + a pre-resolved arm64 venv in `Contents/Resources`.
- **Pros**: real interpreter (no freezer guesswork); relocatable by design; fully
  **offline-capable** first run.
- **Cons**: still ~3 GB in the bundle; still must sign every nested torch dylib with
  hardened runtime + `disable-library-validation`; **every update re-downloads ~3 GB**.

### (C) Lean bootstrap — tiny signed `.app` that installs deps on first run — **chosen**
A small signed/notarized `.app` ships only: the launcher, a signed `uv` binary, the
`app/` source, the ~32 MB vendored diarizer, and the pre-built frontend. On first
launch it uses bundled `uv` to fetch `python-build-standalone` + resolve the arm64
wheel set into `~/Library/Application Support/WhisperX/runtime`, then launches Flask
from there. ffmpeg + ML models also download on first run.
- **Pros**: genuinely **lean** artifact (tens of MB); **smallest signing surface**
  (sign only the launcher + `uv`); **updates stay tiny** (only the small `.app`
  changes — the multi-GB venv + models persist); no freezer fragility.
- **Gatekeeper**: it only enforces on code carrying the `com.apple.quarantine`
  xattr, applied by *download agents* (browsers, Mail). Files an app downloads via
  its own network code (uv's wheels + interpreter) are **not** quarantined, so the
  torch dylibs run without their own Developer ID signature/notarization — **as long
  as the downloaded interpreter is spawned as a separate process** (not `dlopen`'d
  into the hardened-runtime launcher, which would trigger library validation). This
  is how `uv`-managed Python already works on macOS. **Re-validate on a real
  signed+notarized build each macOS release.**
- **Cons**: first run needs network and downloads ~3 GB → a long, failure-prone
  first launch for non-technical users (needs solid progress/retry/disk-space UX).

### (D) Tauri 2 (WKWebView) shell over (C) — **chosen UI layer**
Native window pointing at `127.0.0.1:<port>`; Python runs as a sidecar **outside**
the bundle (the (C) venv).
- **Pros**: best non-technical UX — real `.app` window, native menu/quit; Tauri uses
  the OS WKWebView → single-digit-MB shell (vs Electron's ~150 MB Chromium). SSE +
  htmx work in WKWebView.
- **Cons**: adds a Rust toolchain. **Do NOT use Tauri `externalBin`** to embed the
  Python sidecar (reintroduces full dylib signing + a known notarization bug,
  tauri-apps/tauri#11992). Keep Python in Application Support, so only the Rust shell
  is signed.
- *Escape hatch*: ship (C) with a plain `webbrowser.open` launcher if the Rust
  toolchain is too much (the server already supports this via `WHISPERX_OPEN_BROWSER`).

### Rejected outright
- **(E) BeeWare Briefcase** — project restructure + still bundles the full ~3 GB torch stack.
- **(F) conda `constructor` / `.pkg`** — a `.pkg` wizard contradicts the DMG requirement; notarizing a full conda env is a large signing surface.
- **Mac App Store** — sandboxing is incompatible with spawning a downloaded interpreter, `disable-library-validation`, and the ffmpeg shell-out; downloading a Python runtime is likely rejected.

## Recommended design

**(C) lean bootstrap + (D) Tauri WKWebView shell**, distributed as a signed +
notarized DMG, updated by manual DMG re-download.

```
Whisperx.app  (signed + notarized, tens of MB)
├─ Tauri shell (Rust)             → native window, health-check, loads 127.0.0.1:<port>
├─ uv  (signed binary)            → first-run bootstrap
├─ app/  (source) + static/vendor (pre-built frontend)
└─ app/models/…community-1 (~32 MB vendored diarizer, offline baseline)

~/Library/Application Support/WhisperX/   (OUTSIDE the bundle → survives updates)
├─ runtime/      python-build-standalone + arm64 venv (torch, whisperx, …)
├─ sessions.db   + sessions/<id>/…                    (user state)
├─ models/       refreshed diarization models
├─ hf/           (optional HF_HOME relocation for model weights)
└─ bin/ffmpeg    (downloaded-unquarantined or bundled+signed)
```

**First-run flow:** Tauri shows a setup/progress window → bundled `uv` installs the
runtime into Application Support → ffmpeg provisioned → Flask spawned (separate
process) on `127.0.0.1` → `/healthz` passes → WKWebView loads the UI → Whisper /
wav2vec2 models download lazily on the first transcription (the existing SSE stages
surface this).

**Update flow:** user downloads a new DMG and replaces the app. The runtime, models,
DB, and session files all live in Application Support; the Keychain entry is keyed to
the **frozen Team ID + bundle ID** — so nothing is lost. The new launcher reuses (or
incrementally updates, if the pinned lockfile changed) the existing venv.

## Code changes (already landed in this branch)

These are the approach-independent repo edits the installer depends on:

- **`app/paths.py`** (new) — `data_dir()`: honors `WHISPERX_DATA_DIR`, else macOS
  `~/Library/Application Support/WhisperX`, else the historical `app/data`.
- **`app/server.py`** — `DATA_DIR` uses `paths.data_dir()`; `_load_dotenv()` now also
  reads `<data_dir>/.env` (a writable override location for the packaged app); new
  `/healthz` route (liveness + `models_ready`); binds `127.0.0.1` (not `0.0.0.0`)
  with `_choose_port()` falling back to an ephemeral port when `PORT` is taken;
  writes the bound port to `<data_dir>/runtime-port`; optional `WHISPERX_OPEN_BROWSER=1`
  opens the browser; SIGTERM → graceful `_shutdown()` (executor + DB close, idempotent).
- **`app/diarize_model.py`** — `data_root()` shares `paths.data_dir()` so refreshed
  models follow the relocated data dir.
- **`app/jobs.py`** — `JobQueue.shutdown()` (`executor.shutdown(wait=False)`;
  interrupted jobs are reconciled to `error` on next boot).
- **`app/store.py`** — `SessionStore.close()` to checkpoint WAL on shutdown.

`app/store.py` and `app/secret_store.py` need no further logic change — they already
accept an injected data dir and use the Keychain.

## ML-on-macOS landmines

1. **`disable-library-validation` is mandatory** for any bundled-interpreter path;
   its absence manifests as a **crash on first torch import**, not at launch — so
   smoke tests must run a real transcription. (Less relevant to (C), since the
   interpreter is spawned separately and unsigned — but verify.)
2. **Freeze Team ID + bundle identifier forever** — changing either loses the
   Keychain HF token across updates.
3. **Preserve the existing MPS workaround** — `pipeline._align_device` forces wav2vec2
   alignment to CPU on Apple Silicon to dodge an MPS conv-channel crash; don't move it to MPS.
4. **ffmpeg must be bundled-and-signed or downloaded-unquarantined** — a quarantined
   downloaded ffmpeg is Gatekeeper-blocked on exec.
5. **SQLite WAL must stay on local disk** (Application Support) — never iCloud/network.
6. **Keep the Python runtime out of the Tauri bundle** — `externalBin` + notarization
   is currently buggy; the runtime lives in Application Support.
7. **Re-validate the Gatekeeper/quarantine assumption** (app-downloaded dylibs run
   unsigned) on a clean VM with a real signed+notarized build, every macOS release.

## Verification / proof-of-concept order

The (C) design rests on one OS-behavior assumption, so validate in this order
**before** investing in full Tauri integration:

1. **Code changes** *(done)* — app runs from source with the relocated data dir;
   `127.0.0.1:<port>`, `/healthz`, and graceful SIGTERM all work; relevant tests pass.
2. **Bootstrap PoC (no Tauri yet)** — a minimal signed+notarized launcher `.app` that
   bundles `uv`, on first run resolves the arm64 venv into Application Support, spawns
   Flask as a **separate process**, and `webbrowser.open`s the UI. **Run a real
   transcription** on a **clean macOS VM** with the DMG fetched via a browser (so the
   `.app` is quarantined) to prove app-downloaded dylibs execute without their own
   signature/notarization.
3. **Update preservation test** — create sessions + set an HF token; install a new DMG
   built with the **same Team/bundle ID**; confirm the DB, session files, and Keychain
   token survive and the venv is reused.
4. **Tauri shell** — only after (2)–(3) pass, replace the browser-open launcher with the
   Tauri WKWebView window (health-check gate, native quit → SIGTERM). Verify SSE renders
   in WKWebView.
5. **Failure-path UX** — simulate no-network and low-disk first runs; confirm the setup
   window surfaces a clear, retryable error rather than a silent hang.

## Open follow-ups
- **Bundle ffmpeg** (simpler/offline-safe, +~60 MB) vs **download it** (leaner) —
  recommend bundling+signing for non-technical reliability.
- Relocate **`HF_HOME`** under the data dir (single-folder reset/uninstall) vs leave at
  `~/.cache/huggingface` (models shared across versions for free).
