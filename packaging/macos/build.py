#!/usr/bin/env python3
"""macOS packaging driver for the Manuscript (WhisperX web app) bundle.

This is design (B) from MACOS_INSTALLER.md: a self-contained, signed ``.app`` that
embeds a relocatable Python runtime (python-build-standalone + an arm64 venv with
torch / whisperx / mlx / whispercpp / gdrive / Flask / keyring) plus a bundled,
signed ffmpeg. Only the ML model *weights* download on first run.

The orchestration lives in the Makefile; this file owns the phases where real
control flow matters — assembling the relocatable runtime and the **leaf-first
deep codesign walk** (the part that's fragile in bash).

Subcommands (run via the Makefile, or directly):
    skeleton   build/<App>.app/Contents/{MacOS,Resources} + Info.plist + icon
    runtime    embed python-build-standalone + arm64 venv into Resources
    app        copy app/ source + built frontend + vendored diarizer into Resources
    ffmpeg     fetch/place a static arm64 ffmpeg at Resources/bin/ffmpeg
    launcher   compile launcher.c -> Contents/MacOS/<App>
    sign       deep, leaf-first codesign (IDENTITY env; default ad-hoc "-")
    notarize   notarytool submit + staple (needs a Developer ID Application cert)
    dmg        build a drag-to-Applications DMG
    clean      remove build/

Everything is arm64-only and Apple-Silicon-only by design.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- Frozen identity (see MACOS_INSTALLER.md → "Frozen signing identity") --------
# Changing either of these loses every Keychain secret across updates. Do NOT edit.
BUNDLE_ID = "com.anvil7.manuscript.transcription"
TEAM_ID = "Q8HKVK78G9"

APP_NAME = "Manuscript"
PYTHON_VERSION = os.environ.get("PYTHON_VERSION", "3.12")

# A static arm64 ffmpeg. Overridable via FFMPEG_URL (or FFMPEG_PATH for a local
# binary). The default is a community arm64 static build; pin/verify per release.
FFMPEG_URL = os.environ.get(
    "FFMPEG_URL", "https://www.osxexperts.net/ffmpeg711arm.zip"
)

# --- Paths -----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # packaging/macos
REPO = HERE.parent.parent                        # repo root
BUILD = HERE / "build"
APP = BUILD / f"{APP_NAME}.app"
CONTENTS = APP / "Contents"
MACOS = CONTENTS / "MacOS"
RES = CONTENTS / "Resources"
RUNTIME = RES / "runtime"                         # the venv
PY_HOME = RES / "python"                          # the embedded interpreter
ENTITLEMENTS = HERE / "entitlements.plist"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def version() -> str:
    """App version: WHISPERX_VERSION env, else the pyproject project.version."""
    if v := os.environ.get("WHISPERX_VERSION"):
        return v
    import re

    txt = (REPO / "pyproject.toml").read_text()
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', txt)
    return m.group(1) if m else "0.0.0"


# --- skeleton --------------------------------------------------------------------
def cmd_skeleton(_args) -> None:
    MACOS.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    icns = REPO / "app" / "branding" / f"{APP_NAME}.icns"
    if icns.exists():
        shutil.copy2(icns, RES / f"{APP_NAME}.icns")
    else:
        print(f"  ! {icns} missing — bundle will have no icon")

    ver = version()
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": f"{APP_NAME}.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": ver,
        "CFBundleVersion": ver,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "12.0",
        "LSArchitecturePriority": ["arm64"],
        "NSHighResolutionCapable": True,
        # Server + browser; no docs. Keep a normal (non-agent) app for the PoC.
        "LSApplicationCategoryType": "public.app-category.productivity",
    }
    with open(CONTENTS / "Info.plist", "wb") as fh:
        plistlib.dump(info, fh)
    print(f"  wrote {CONTENTS/'Info.plist'} (version {ver})")


# --- runtime ---------------------------------------------------------------------
def _managed_python_root() -> Path:
    """Install (if needed) and locate the uv-managed python-build-standalone root."""
    run(["uv", "python", "install", PYTHON_VERSION])
    out = subprocess.run(
        ["uv", "python", "find", PYTHON_VERSION],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # out = .../cpython-X-macos-aarch64-none/bin/python3  -> root is two up
    root = Path(out).resolve().parent.parent
    if not (root / "bin").exists():
        sys.exit(f"unexpected managed python layout: {out}")
    return root


def cmd_runtime(_args) -> None:
    if PY_HOME.exists():
        shutil.rmtree(PY_HOME)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)

    # 1) Embed the interpreter itself (python-build-standalone is relocatable).
    src = _managed_python_root()
    print(f"  copying interpreter {src} -> {PY_HOME}")
    shutil.copytree(src, PY_HOME, symlinks=True)

    # 2) A relocatable venv built FROM the embedded interpreter, so venv (runtime/)
    #    and interpreter (python/) sit side by side in Resources and move together.
    run(["uv", "venv", "--relocatable", "--python",
         PY_HOME / "bin" / "python3", RUNTIME])

    venv_py = RUNTIME / "bin" / "python"
    # 3) Install the project (non-editable, so it's copied in) + Mac ASR backends +
    #    cloud-backup deps + the app-only runtime deps (Flask/keyring live in
    #    app/requirements.txt, not pyproject — see app/start.sh).
    run([
        "uv", "pip", "install", "--python", venv_py,
        f"{REPO}[mlx,whispercpp,gdrive]", "Flask>=3.0", "keyring>=24",
    ])
    # 4) uv records the interpreter as an ABSOLUTE symlink (we passed an absolute
    #    --python). codesign rejects absolute symlinks in a bundle, and they'd
    #    break once the .app moves. Rewrite every in-bundle symlink as relative.
    _relativize_symlinks(APP)
    print("  runtime ready")


def _relativize_symlinks(root: Path) -> None:
    """Rewrite absolute symlinks whose target is inside the bundle as relative.

    Gatekeeper/codesign require bundle symlinks to be relative and stay within the
    bundle; an absolute target also pins the venv to its build path. Anything that
    points *outside* the bundle is a hard error (it can't ship)."""
    app_root = root.resolve()
    fixed = 0
    # followlinks=False (default): symlinked dirs appear in dirnames, not descended.
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            link = Path(dirpath) / name
            if not link.is_symlink():
                continue
            target = os.readlink(link)
            if not os.path.isabs(target):
                continue
            resolved = Path(os.path.normpath(target)).resolve()
            try:
                resolved.relative_to(app_root)
            except ValueError:
                sys.exit(f"symlink escapes bundle, cannot ship: {link} -> {target}")
            rel = os.path.relpath(resolved, link.parent)
            link.unlink()
            link.symlink_to(rel)
            fixed += 1
    print(f"  relativized {fixed} in-bundle symlink(s)")


# --- app source ------------------------------------------------------------------
def cmd_app(_args) -> None:
    # Ensure the frontend bundle exists (app/static/vendor); build with bun if not.
    vendor = REPO / "app" / "static" / "vendor"
    if not vendor.exists():
        print("  building frontend (bun run build)…")
        run(["bun", "install"], cwd=REPO / "app")
        run(["bun", "run", "build"], cwd=REPO / "app")

    dest = RES / "app"
    if dest.exists():
        shutil.rmtree(dest)

    # Copy app/ but skip dev-only / regenerable / user-state dirs.
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "data", "node_modules", "tests",
        "src", "*.test.ts", "bunfig.toml", "bun.lock", "package.json",
        "build.ts", "tsconfig.json",
    )
    shutil.copytree(REPO / "app", dest, ignore=ignore)
    # Drop any committed .env so packaged runs start clean (data-dir .env still wins).
    (dest / ".env").unlink(missing_ok=True)
    print(f"  copied app/ -> {dest} (vendored diarizer + frontend included)")


# --- ffmpeg ----------------------------------------------------------------------
def cmd_ffmpeg(_args) -> None:
    bindir = RES / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    target = bindir / "ffmpeg"

    local = os.environ.get("FFMPEG_PATH")
    if local:
        shutil.copy2(local, target)
    else:
        print(f"  downloading ffmpeg from {FFMPEG_URL}")
        tmp = BUILD / "ffmpeg-download"
        tmp.mkdir(exist_ok=True)
        archive = tmp / "ffmpeg.zip"
        urllib.request.urlretrieve(FFMPEG_URL, archive)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                # find the 'ffmpeg' member regardless of nesting
                member = next(n for n in zf.namelist()
                              if Path(n).name == "ffmpeg")
                with zf.open(member) as s, open(target, "wb") as d:
                    shutil.copyfileobj(s, d)
        else:
            shutil.copy2(archive, target)

    target.chmod(0o755)
    # Sanity: must be a Mach-O arm64 binary.
    out = subprocess.run(["file", str(target)], capture_output=True, text=True).stdout
    print(f"  {out.strip()}")
    if "arm64" not in out:
        print("  ! WARNING: bundled ffmpeg is not arm64 — replace via FFMPEG_PATH/URL")


# --- launcher --------------------------------------------------------------------
def cmd_launcher(_args) -> None:
    MACOS.mkdir(parents=True, exist_ok=True)
    out = MACOS / APP_NAME
    # major.minor only (e.g. 3.12) -> pythonX.Y tag for the site-packages path.
    pytag = "python" + ".".join(PYTHON_VERSION.split(".")[:2])
    run(["clang", "-arch", "arm64", "-O2",
         f'-DPYTAG="{pytag}"', "-o", out, HERE / "launcher.c"])
    out.chmod(0o755)
    print(f"  built launcher -> {out} (PYTAG={pytag})")


# --- sign ------------------------------------------------------------------------
def _is_macho(p: Path) -> bool:
    """True if p starts with a Mach-O or universal magic number."""
    try:
        with open(p, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return False
    return magic in (
        b"\xcf\xfa\xed\xfe",  # MH_MAGIC_64 (little-endian)
        b"\xfe\xed\xfa\xcf",  # MH_MAGIC_64 (big-endian)
        b"\xca\xfe\xba\xbe",  # FAT_MAGIC (universal)
        b"\xbe\xba\xfe\xca",  # FAT_CIGAM
    )


def _macho_targets() -> list[Path]:
    """Every Mach-O file under the bundle, leaf-first (deepest path first).

    Signing inner dylibs/.so before the bundles that contain them is what keeps
    notarization happy; sorting by path depth descending gives that order.
    """
    found: list[Path] = []
    for root, _dirs, files in os.walk(APP):
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                continue
            if p.suffix in (".dylib", ".so") or _is_macho(p):
                found.append(p)
    found.sort(key=lambda p: len(p.parts), reverse=True)
    return found


def cmd_sign(_args) -> None:
    identity = os.environ.get("IDENTITY", "-")  # "-" = ad-hoc (local PoC)
    adhoc = identity == "-"
    base = ["codesign", "--force", "--options", "runtime",
            "--entitlements", str(ENTITLEMENTS), "--sign", identity]
    if not adhoc:
        base += ["--timestamp"]          # secure timestamp needs a real identity
    else:
        print("  IDENTITY unset → ad-hoc signing (local smoke-test only; "
              "NOT notarizable). Set IDENTITY='Developer ID Application: …'.")

    targets = _macho_targets()
    print(f"  signing {len(targets)} Mach-O files leaf-first…")
    for p in targets:
        run(base + [str(p)])

    # The bundle (main executable + Info.plist seal) is signed last.
    run(base + [str(APP)])

    # Verify the seal (skip strict notarization checks for ad-hoc).
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(APP)])
    print("  signed + verified")


# --- notarize --------------------------------------------------------------------
def cmd_notarize(_args) -> None:
    profile = os.environ.get("NOTARY_PROFILE")
    if not profile:
        sys.exit(
            "Set NOTARY_PROFILE to a notarytool keychain profile. Create one with:\n"
            "  xcrun notarytool store-credentials NOTARY_PROFILE \\\n"
            f"    --apple-id <id> --team-id {TEAM_ID} --password <app-specific-pw>\n"
            "Requires a 'Developer ID Application' certificate (not yet present)."
        )
    zip_path = BUILD / f"{APP_NAME}.zip"
    run(["ditto", "-c", "-k", "--keepParent", str(APP), str(zip_path)])
    run(["xcrun", "notarytool", "submit", str(zip_path),
         "--keychain-profile", profile, "--wait"])
    run(["xcrun", "stapler", "staple", str(APP)])
    print("  notarized + stapled")


# --- dmg -------------------------------------------------------------------------
def cmd_dmg(_args) -> None:
    dmg = BUILD / f"{APP_NAME}-{version()}-arm64.dmg"
    staging = BUILD / "dmg"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    run(["cp", "-R", str(APP), str(staging / f"{APP_NAME}.app")])
    os.symlink("/Applications", staging / "Applications")
    dmg.unlink(missing_ok=True)
    run(["hdiutil", "create", "-volname", APP_NAME, "-srcfolder", str(staging),
         "-ov", "-format", "UDZO", str(dmg)])
    print(f"  built {dmg}")


def cmd_clean(_args) -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    print("  cleaned build/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("skeleton", "runtime", "app", "ffmpeg", "launcher",
                 "sign", "notarize", "dmg", "clean"):
        sub.add_parser(name).set_defaults(fn=globals()[f"cmd_{name}"])
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
