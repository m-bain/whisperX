# Packaging the WhisperX web app as a macOS installer — research

This document is design research for shipping the web frontend (`app/`) as a
**redistributable macOS app a non-technical user can double-click**, and for
**updating that app without destroying user state** (the SQLite session DB,
per-session audio/transcripts, and the Keychain secrets).

It records the constraints, the packaging options with pros/cons, the
recommended design, and the supporting code changes that have already landed.
The macOS-specific build steps (Tauri shell, code signing, notarization, DMG)
must be executed on a Mac with an Apple Developer ID — they are documented here
but not yet implemented.

### Frozen signing identity (do not ever change)
Keychain ACLs bind to the signing identity, so these are frozen **forever** —
changing either loses every stored secret across updates:

- **Bundle identifier**: `com.anvil7.manuscript.transcription`
- **Apple Team ID**: `Q8HKVK78G9`

## Requirements

| Decision | Choice |
|---|---|
| Audience / launch | **Non-technical, double-click `.app`** (drag-to-Applications DMG). No terminal, Homebrew, or pre-installed Python. |
| Signing | **Signed + notarized** with an Apple Developer ID. Gatekeeper must open cleanly. |
| Architecture | **Apple Silicon (arm64) only.** Ships **both** Mac ASR backends implemented in `pipeline.py` — `whispercpp` (Metal, the Apple-Silicon default) **and** `mlx` — so users can switch in the device picker. |
| Bundling | **Self-contained** (design (B)): the full arm64 venv (torch + whisperx + `whispercpp`/`mlx`) and **bundled+signed ffmpeg** ship inside the signed `.app` (~3 GB DMG). Only the **ML model weights** download on first run. |
| Window | **Native desktop window** (Tauri / WKWebView), not "just a browser tab". |
| Updates | **Manual re-download of a new DMG.** No Sparkle/auto-update for v1. |

### State that MUST survive an update
- **SQLite DB + session files** — `sessions.db` (WAL mode) and `sessions/<id>/` (audio, transcripts).
- **Refreshed diarization models** — under `<data_dir>/models`.
- **Keychain secrets** — macOS Keychain via `keyring`, all under service `manuscript-whisperx`
  (see `app/secret_store.py`):
  - `hf_token` — Hugging Face token.
  - `google_drive_creds` — Google Drive OAuth credentials (refresh token) for cloud backup.
  - `google_drive_folder` — chosen Drive backup folder name. Deliberately in the
    Keychain rather than the data dir, because the data dir is itself mirrored to Drive.

All Keychain entries survive automatically **iff** every release keeps the **same
Team ID and bundle identifier** (recorded above; ACLs are bound to the signing
identity). The DB and files survive **iff** they live *outside* the `.app` bundle —
hence the data dir is now relocated (see code changes below). Model **weights** live
at `~/.cache/huggingface` (see Resolved decisions) so they too survive an update.

**Complementary off-machine path — cloud backup (`app/backup/`).** The app can now
mirror the data dir to Google Drive (snapshot-under-lock DB copy + incremental,
content-addressed session-file sync; local is authoritative, remote is a one-way
mirror). This is *not* a substitute for keeping state outside the bundle, but it is
a second recovery/migration path: a fresh install on a new Mac can restore from
Drive. Backup status/connect flow already streams over the `/backup/*` SSE channels.

## Options considered

torch (~2.5–3.5 GB on disk, arm64 CPU/MPS) is an **in-process import** — it can't
be deferred the way ffmpeg and model weights can. So the real choice is *where the
torch stack lives*, which decides the code-signing surface and the update size.

torch is required even on Apple Silicon: ASR runs on whisper.cpp/Metal or MLX, but
the **VAD / align / diarize stages remain torch** (on `mps`, alignment forced to
`cpu` — see landmine 3). The venv therefore carries torch **plus** both the
`whispercpp` and `mlx` extras. **Decision: bundle this whole stack inside the signed
`.app` (design (B)).** This trades the "lean / tiny update" goal for a self-contained
artifact that needs no runtime download and removes the app-downloaded-dylib
Gatekeeper gamble entirely — at the cost of a ~3 GB DMG and a ~3 GB replace on every
update.

### (A) Frozen `.app` — PyInstaller / py2app — **rejected**
- **Pros**: one self-contained artifact; well-known signing recipe.
- **Cons**: whisperx's lazy/dynamic imports + data files (pyannote/nltk/VAD assets)
  force brittle `hiddenimports`/`collect-all` lists; the freezer **rewrites dylib load
  paths, invalidating the ad-hoc signatures inside the torch/ctranslate2 wheels**, so
  the re-sign step also has to repair load commands; slow notarization. The
  load-path rewriting is what makes (A) brittle vs (B) — (B) keeps wheel layout
  intact, so signing is mechanical.

### (B) Embedded relocatable interpreter inside the signed `.app` — **chosen**
Ship `python-build-standalone` + a pre-resolved arm64 venv (torch, whisperx,
`whispercpp`, `mlx`) and a signed ffmpeg in `Contents/Resources`. Flask is spawned
as a **separate process** from that bundled interpreter.
- **Pros**: real interpreter (no freezer guesswork); relocatable by design; **fully
  offline-capable except model-weight download**; **no Gatekeeper/quarantine gamble**
  (everything ships signed + notarized); first run is fast and robust — no multi-GB
  download to fail on. Because the wheel layout is untouched (unlike (A)), the dylib
  load paths stay valid and signing is a mechanical deep-sign.
- **Cons**: ~3 GB in the bundle; **must sign every nested torch dylib** with hardened
  runtime + `disable-library-validation` (see landmine 1); **every update re-downloads
  ~3 GB** DMG. Accepted: self-containment + reliability for non-technical users beats
  update size here.

### (C) Lean bootstrap — tiny signed `.app` that installs deps on first run — **rejected**
A small `.app` shipping only the launcher + a signed `uv`, fetching
`python-build-standalone` + the arm64 wheel set into Application Support on first run.
- **Why rejected**: lean artifact + tiny updates, but the first run needs network and
  downloads ~3 GB → a long, failure-prone first launch for non-technical users; and it
  **rests on the unverified assumption** that app-downloaded (unquarantined) torch
  dylibs execute without their own signature/notarization. We chose to **not bet on
  that** and bundle+sign instead (= (B)).

### (D) Tauri 2 (WKWebView) shell over (B) — **chosen UI layer**
Native window pointing at `127.0.0.1:<port>`; the Python server runs as a child
process spawned from the **in-bundle** interpreter (`Contents/Resources`).
- **Pros**: best non-technical UX — real `.app` window, native menu/quit; Tauri uses
  the OS WKWebView → single-digit-MB shell (vs Electron's ~150 MB Chromium). SSE +
  htmx work in WKWebView.
- **Cons**: adds a Rust toolchain. **Do NOT use Tauri `externalBin`** for the Python
  runtime (a known notarization bug, tauri-apps/tauri#11992); place the interpreter in
  `Contents/Resources` yourself and `spawn` it — the Rust shell and the whole Resources
  tree are signed in one pass (landmine 6).
- **Sequencing**: ship a plain `webbrowser.open` launcher **first** as the bootstrap
  PoC (the server already supports this via `WHISPERX_OPEN_BROWSER`), then add the
  Tauri shell once the bundle + signing pipeline is proven.

### Rejected outright
- **(E) BeeWare Briefcase** — project restructure + still bundles the full ~3 GB torch stack.
- **(F) conda `constructor` / `.pkg`** — a `.pkg` wizard contradicts the DMG requirement; notarizing a full conda env is a large signing surface.
- **Mac App Store** — sandboxing is incompatible with spawning a downloaded interpreter, `disable-library-validation`, and the ffmpeg shell-out; downloading a Python runtime is likely rejected.

## Recommended design

**(B) embedded signed interpreter + (D) Tauri WKWebView shell**, distributed as a
signed + notarized DMG (~3 GB), updated by manual DMG re-download.

```
Whisperx.app  (signed + notarized, ~3 GB)
└─ Contents/
   ├─ MacOS/<tauri-shell>           → native window, health-check, loads 127.0.0.1:<port>
   └─ Resources/                    (signed in one deep-sign pass; hardened runtime)
      ├─ runtime/   python-build-standalone + arm64 venv (torch, whisperx, whispercpp, mlx)
      ├─ bin/ffmpeg                  (bundled + signed)
      ├─ app/  (source) + static/vendor (pre-built frontend)
      └─ app/models/…community-1     (~31 MB vendored diarizer, offline baseline)

~/Library/Application Support/WhisperX/   (OUTSIDE the bundle → survives updates)
├─ sessions.db   + sessions/<id>/…        (user state)
├─ models/       refreshed diarization models
└─ .env, runtime-port                     (writable overrides + bound-port handoff)

~/.cache/huggingface/                     (model weights — shared across versions, survives update)
macOS Keychain (svc manuscript-whisperx): hf_token, google_drive_creds, google_drive_folder
```

**First-run flow:** Tauri launches → spawns Flask (separate process) from the in-bundle
interpreter on `127.0.0.1` → `/healthz` passes → WKWebView loads the UI. No runtime
download. Whisper / wav2vec2 model **weights** download lazily on the first
transcription (the existing SSE stages surface this); everything else is already
present, so first run works offline up to that point.

**Update flow:** user downloads a new DMG and replaces the app. The bundled venv +
ffmpeg are replaced wholesale (no stale-venv reconciliation needed). DB, session
files, refreshed models (Application Support), model weights (`~/.cache/huggingface`),
and the Keychain secrets (keyed to the **frozen Team ID + bundle ID** above) all live
outside the bundle — so nothing is lost.

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

## Build tooling (`packaging/macos/`)

The (B) pipeline is scaffolded and the **browser-open PoC builds + runs**:

- **`build.py`** — phase driver: `skeleton` (`.app` tree + `Info.plist` + icon),
  `runtime` (embed interpreter + venv), `app` (copy source + frontend + diarizer +
  bake `defaults.env`), `ffmpeg`, `launcher`, `sign` (leaf-first deep codesign),
  `notarize`, `dmg`. Frozen `BUNDLE_ID`/`TEAM_ID` are env-overridable
  (`MANUSCRIPT_BUNDLE_ID`/`MANUSCRIPT_TEAM_ID`) with the frozen values as defaults.
- **`Makefile`** — orchestrator; `make poc` (ad-hoc, local) and `make release`
  (`IDENTITY=…` + `NOTARY_PROFILE=…`). Per-phase targets avoid rebuilding the runtime.
  `NO_TIMESTAMP=1` skips the per-file secure timestamp for fast test signing.
- **`launcher.c`** — tiny signed Mach-O `CFBundleExecutable`; resolves the bundle,
  sets env (`WHISPERX_OPEN_BROWSER=1`, `PYTHONPATH`, bundled ffmpeg on `PATH`),
  `exec`s the interpreter (`-m app.server`). When not attached to a terminal
  (Finder/`open`) it redirects stdout/stderr to `<data dir>/manuscript.log` so logs
  survive; a direct terminal run still streams (the `isatty` check).
- **`entitlements.plist`** — hardened-runtime: `disable-library-validation` (+ jit /
  unsigned-exec-mem / dyld-env).
- **`bundle-defaults.env`** (gitignored) — ship-with-the-app config baked into the
  bundle as `Resources/app/defaults.env`: `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`
  (Drive OAuth app) + `WHISPERX_BACKUP_BACKEND=gdrive`. **Not** per-user secrets
  (`HF_TOKEN` and the Drive refresh token stay in the Keychain). Loaded at lowest
  precedence so a user's `<data dir>/.env` or real env vars override it.

**Config precedence** (highest first; `server.py` loads each, never overriding an
already-set key): real env → `app/.env` (dev) → `<data dir>/.env` (per-machine) →
`app/defaults.env` (bundled ship-defaults).

Resulting bundle: **~1.4 GB** (`python` 60 MB + `runtime` venv 1.3 GB + `app` 56 MB) —
smaller than the ~3 GB estimate (the arm64 mac torch wheel is CPU/MPS-only).

### Confirmed working (validated on Apple Silicon)

The PoC bundle has been exercised end-to-end and is **functionally complete** for
local/internal testing (everything except notarized external distribution):

- **Builds, signs, relocates, serves** — `make poc`; a relocated copy (original
  hidden) serves `/healthz` and the UI.
- **Real-signature signing** — also signed with an **Apple Development** cert under
  the real `TeamIdentifier=Q8HKVK78G9` + hardened runtime. torch / ctranslate2 /
  pyannote all import under it → **`disable-library-validation` confirmed under a
  real (non-ad-hoc) signature** (landmine 1; still re-check on a Developer ID build).
- **Transcription** — confirmed end-to-end via the GUI (ASR + alignment). whisper.cpp
  (Metal) verified; MLX selectable.
- **Cloud backup** — confirmed end-to-end via the GUI: the backend comes from the
  **bundled `defaults.env`** alone (no `app/.env` in the bundle, no `<data dir>/.env`),
  the Drive OAuth link + refresh token live in the **Keychain** and persist across
  rebuilds, and a real backup runs/uploads.
- **Logging** — Finder/`open` launches write `<data dir>/manuscript.log`.

### Findings from the PoC build

1. **The uv venv does NOT relocate.** `uv venv --relocatable` still writes an
   **absolute** `pyvenv.cfg home`, and CPython ignores a relative one — so invoking
   the venv's `bin/python` from a moved/renamed bundle fails with `No module named
   'encodings'`. The **base python-build-standalone interpreter relocates fine**, so
   the launcher runs `Resources/python/bin/python3` directly with the venv's
   `site-packages` on `PYTHONPATH` (the venv is still where deps are installed; we
   just don't depend on its broken home resolution). Verified: relocated copy with the
   original hidden serves `/healthz` and transcribes.
2. **`codesign` rejects absolute symlinks in the bundle.** `build.py` relativizes all
   in-bundle symlinks (and hard-fails on any that escape the bundle) before signing.
3. **torchcodec dylibs fail to load** — `libtorchcodec_core*.dylib` want
   `@rpath/libavutil.*.dylib` (FFmpeg *shared libs*, which we don't bundle — we bundle
   the ffmpeg *binary*). **Non-fatal:** whisperx decodes audio via the ffmpeg CLI, so
   the warning is cosmetic; a full ASR+align transcription completes. Follow-up:
   suppress the warning, or bundle the FFmpeg dylibs + fix rpaths if any path ever
   needs torchcodec.
4. **No `Developer ID Application` cert yet** — the installed certs (Apple Development,
   iPhone Distribution) can't notarize a macOS app. Real sign + notarize + the
   clean-VM Gatekeeper run + the Developer-ID re-check of landmine 1 are **blocked**
   until that cert is created in team `Q8HKVK78G9`.

## ML-on-macOS landmines

1. **`disable-library-validation` is mandatory** — the chosen (B) design bundles the
   interpreter, so this entitlement is **required**. Its absence manifests as a
   **crash on first torch import**, not at launch — so smoke tests must run a real
   transcription, not just open the window.
2. **Freeze Team ID + bundle identifier forever** — changing either loses *all*
   Keychain secrets across updates (`hf_token`, `google_drive_creds`, `google_drive_folder`).
3. **Preserve the existing MPS workaround** — on the Apple-Silicon backends
   (`mlx`/`whispercpp`) the torch stages run on `mps`, but `pipeline._align_device`
   forces **wav2vec2 alignment to CPU** to dodge an MPS conv-channel crash on large
   wav2vec2 models; don't move alignment to MPS.
4. **ffmpeg is bundled-and-signed** (decided) — it ships in `Contents/Resources/bin`
   and is deep-signed with the rest of Resources. (A *downloaded* ffmpeg would be
   Gatekeeper-blocked on exec if quarantined — avoided by bundling.)
5. **SQLite WAL must stay on local disk** (Application Support) — never iCloud/network.
6. **Don't use Tauri `externalBin` for the runtime** — `externalBin` + notarization is
   currently buggy (tauri-apps/tauri#11992). The interpreter still lives **in** the
   bundle (`Contents/Resources`, per (B)) — just place + spawn it yourself rather than
   via the `externalBin` mechanism, and deep-sign Resources in one pass.

## Verification / proof-of-concept order

The (B) design's load-bearing risk is the deep-sign + hardened-runtime torch import,
so validate in this order **before** investing in full Tauri integration:

1. **Code changes** *(done)* — app runs from source with the relocated data dir;
   `127.0.0.1:<port>`, `/healthz`, and graceful SIGTERM all work; relevant tests pass.
2. **Bundle + browser-open PoC (no Tauri yet)** *(done, ad-hoc; notarization pending
   cert)* — `packaging/macos/make poc` builds the bundle (`python-build-standalone` +
   venv with torch/whisperx/`whispercpp`/`mlx`/`gdrive`/Flask/keyring + signed ffmpeg),
   deep-signs with hardened runtime + `disable-library-validation`, and ships an `.app`
   that spawns Flask from the in-bundle interpreter and `webbrowser.open`s the UI.
   **Verified ad-hoc:** a relocated copy (original hidden) serves `/healthz` and runs a
   full ASR+align transcription — proving torch/ctranslate2/pyannote load under the
   hardened runtime (landmine 1, indicative). **Still required (cert-blocked):** real
   `Developer ID Application` sign + notarize, the **clean-VM quarantined run**, the
   Developer-ID re-check of landmine 1, and confirming the `mlx` backend end-to-end.
3. **Update preservation test** — create sessions + set an HF token + link Drive backup;
   install a new DMG built with the **same Team/bundle ID**; confirm the DB, session
   files, `~/.cache/huggingface` weights, and all three Keychain secrets survive.
4. **Tauri shell** — only after (2)–(3) pass, replace the browser-open launcher with the
   Tauri WKWebView window (health-check gate, native quit → SIGTERM). Verify SSE renders
   in WKWebView; re-run the deep-sign/notarize with the Rust shell included.
5. **Failure-path UX** — simulate low-disk and offline first transcription (model-weight
   download fails); confirm the UI surfaces a clear, retryable error rather than a
   silent hang.

## Resolved decisions
- **ffmpeg**: **bundle + sign** (+~60 MB) — offline-safe, no Gatekeeper-on-exec risk.
- **`HF_HOME`**: **leave at `~/.cache/huggingface`** — weights shared across versions
  and survive a data-dir reset for free. Trade-off accepted: uninstall leaves a stray
  cache, and the weights are **not** in the Drive backup (re-downloaded on a fresh
  machine — acceptable, they're public).
- **Torch stack**: **bundle + sign inside the `.app`** (design (B)), not the lean
  first-run bootstrap (C). Reverses the original "lean / tiny update" goal in favor of
  a self-contained, network-light, Gatekeeper-safe artifact.
- **Mac ASR backends**: ship **both** `whispercpp` (default) and `mlx`.
- **UI shell sequencing**: **browser-open PoC first**, Tauri after the bundle/signing
  pipeline is proven.
- **Identity**: bundle ID `com.anvil7.manuscript.transcription`, Team ID `Q8HKVK78G9`
  — frozen forever.

## Residual risks (accepted, no open decision)
- **~3 GB per update.** Inherent to (B); no Sparkle/delta updates in v1.
- **Hardened-runtime torch import** must be smoke-tested on every macOS release with a
  real transcription (landmine 1) — a *verification* task, not an open question.
