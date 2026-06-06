# macOS packaging — Manuscript (WhisperX web app)

Build tooling for the signed, self-contained macOS `.app` described in
[`../../MACOS_INSTALLER.md`](../../MACOS_INSTALLER.md) (**design (B)**: the full
arm64 Python runtime + ffmpeg ship *inside* the signed bundle; only ML model
weights download on first run).

The bundle's `CFBundleExecutable` is a native **Tauri 2 (WKWebView) shell**
(`tauri/`, a plain `cargo build` library-mode binary — not `cargo tauri build`):
it spawns the embedded interpreter, gates on `/healthz`, and opens the UI in a
native desktop window. `build.py` copies the compiled binary into the bundle and
remains the single owner of the `.app` tree + deep-sign.

## Layout produced

```
build/Manuscript.app/Contents/
├─ MacOS/Manuscript            native Tauri WKWebView shell (tauri/src/main.rs)
├─ Info.plist                  CFBundleIdentifier = com.anvil7.manuscript.transcription
└─ Resources/
   ├─ python/                  embedded python-build-standalone interpreter
   ├─ runtime/                 relocatable venv (torch, whisperx, mlx, whispercpp, gdrive, Flask, keyring)
   ├─ app/                     app/ source + built frontend + vendored diarizer
   ├─ bin/ffmpeg               bundled, signed static arm64 ffmpeg
   └─ Manuscript.icns
```

User state stays **outside** the bundle (survives updates): `~/Library/Application
Support/WhisperX/` (DB, sessions, refreshed models), `~/.cache/huggingface`
(weights), and the macOS Keychain (`hf_token`, `google_drive_creds`,
`google_drive_folder`).

## Prerequisites

- Apple Silicon Mac, Xcode command-line tools (`codesign`, `ditto`, `hdiutil`,
  and for release `notarytool`/`stapler`).
- A Rust toolchain (`cargo`) for the Tauri shell.
- [`uv`](https://docs.astral.sh/uv/) and [`bun`](https://bun.sh) on `PATH`.

## Build

```bash
# Local PoC: ad-hoc signed, launches on THIS machine (not notarizable).
make poc
open build/Manuscript.app
# Server logs land in ~/Library/Application Support/WhisperX/manuscript.log
```

The first `make poc` resolves the ~3 GB runtime — slow. Per-phase targets
(`make runtime`, `make app`, …) let you redo one step without that cost.

## Release (signed + notarized + DMG)

> **Blocked until a `Developer ID Application` certificate exists** in the Anvil7
> team (`Q8HKVK78G9`). The certs currently installed (Apple Development, iPhone
> Distribution) **cannot** notarize a macOS app. Create the Developer ID
> Application cert at developer.apple.com → Certificates, then:

```bash
# One-time: store notarization credentials in a keychain profile.
xcrun notarytool store-credentials manuscript-notary \
  --apple-id <apple-id> --team-id Q8HKVK78G9 --password <app-specific-password>

make release \
  IDENTITY="Developer ID Application: Anvil7 UG (haftungsbeschrankt) (Q8HKVK78G9)" \
  NOTARY_PROFILE=manuscript-notary
```

`make release` runs: assemble → leaf-first deep codesign (hardened runtime +
`entitlements.plist`) → notarytool submit `--wait` → staple → DMG.

## Knobs

| Env | Default | Meaning |
|-----|---------|---------|
| `IDENTITY` | `-` (ad-hoc) | codesign identity; set to the Developer ID for release |
| `NOTARY_PROFILE` | – | notarytool keychain profile (required for `notarize`) |
| `PYTHON_VERSION` | `3.12` | embedded interpreter version |
| `FFMPEG_URL` | osxexperts arm64 build | where to fetch the static ffmpeg |
| `FFMPEG_PATH` | – | use a local ffmpeg binary instead of downloading |
| `WHISPERX_VERSION` | pyproject `version` | version stamped into Info.plist / DMG name |

## Smoke test (mandatory per release)

Ad-hoc/notarized, the hardened-runtime torch import only fails **at first
transcription** (landmine 1). After `make poc`, open the app and **run a real
transcription on a short clip** with both the `whisper.cpp` and `mlx` backends
(Settings → Compute Device) before trusting a build.
