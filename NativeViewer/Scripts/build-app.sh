#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGURATION="${CONFIGURATION:-debug}"
TRIPLE="$(swift build --package-path "$ROOT" --configuration "$CONFIGURATION" --show-bin-path)"
APP_DIR="$ROOT/.build/TranscriptViewer.app"

mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$ROOT/Resources/Info.plist" "$APP_DIR/Contents/Info.plist"
cp "$TRIPLE/TranscriptViewer" "$APP_DIR/Contents/MacOS/TranscriptViewer"
chmod +x "$APP_DIR/Contents/MacOS/TranscriptViewer"
codesign --force --deep --sign - "$APP_DIR" >/dev/null

echo "$APP_DIR"
