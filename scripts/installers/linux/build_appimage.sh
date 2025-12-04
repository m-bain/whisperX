#!/bin/bash
# Build Linux AppImage for WhisperX SmartVoice
# Prerequisites:
#   - Python 3.9+ with PyInstaller installed
#   - appimagetool (download from https://github.com/AppImage/AppImageKit)
#   - UV package manager (optional)

set -e  # Exit on error

echo "============================================"
echo "WhisperX SmartVoice Linux AppImage Build"
echo "============================================"
echo

# Change to project root
cd "$(dirname "$0")/../../.."
PROJECT_ROOT=$(pwd)

echo "Project root: $PROJECT_ROOT"
echo

# Build version
VERSION="3.4.2"

echo "Step 1: Building Launcher executable..."
echo "----------------------------------------"
python3 -m PyInstaller launcher.spec --clean
echo "Launcher built successfully!"
echo

echo "Step 2: Building SmartVoice main executable..."
echo "----------------------------------------------"
python3 -m PyInstaller smartvoice.spec --clean
echo "SmartVoice built successfully!"
echo

echo "Step 3: Creating AppImage structure..."
echo "--------------------------------------"

# Create AppDir structure
APPDIR="$PROJECT_ROOT/build/SmartVoice.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR"

# Create directory structure
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/lib"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy launcher
cp -r "$PROJECT_ROOT/dist/SmartVoiceLauncher" "$APPDIR/usr/lib/"

# Copy main application
cp -r "$PROJECT_ROOT/dist/SmartVoice" "$APPDIR/usr/lib/"

# Create launcher script in bin
cat > "$APPDIR/usr/bin/smartvoice-launcher" << 'EOF'
#!/bin/bash
# SmartVoice Launcher wrapper script
APPDIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
exec "$APPDIR/usr/lib/SmartVoiceLauncher/SmartVoiceLauncher" "$@"
EOF

chmod +x "$APPDIR/usr/bin/smartvoice-launcher"

# Create AppRun script
cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
# Main AppRun script for SmartVoice AppImage
APPDIR="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$APPDIR/usr/lib:$LD_LIBRARY_PATH"
exec "$APPDIR/usr/bin/smartvoice-launcher" "$@"
EOF

chmod +x "$APPDIR/AppRun"

# Create desktop file
cat > "$APPDIR/usr/share/applications/smartvoice.desktop" << EOF
[Desktop Entry]
Type=Application
Name=SmartVoice
Comment=WhisperX Automatic Speech Recognition
Exec=smartvoice-launcher
Icon=smartvoice
Categories=AudioVideo;Audio;Utility;
Terminal=false
EOF

# Copy desktop file to AppDir root (required by AppImage)
cp "$APPDIR/usr/share/applications/smartvoice.desktop" "$APPDIR/"

# Create icon (placeholder - replace with actual icon)
# For now, create a simple icon or copy if exists
if [ -f "$PROJECT_ROOT/whisperx/launcher/resources/icon.png" ]; then
    cp "$PROJECT_ROOT/whisperx/launcher/resources/icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/smartvoice.png"
    cp "$PROJECT_ROOT/whisperx/launcher/resources/icon.png" "$APPDIR/smartvoice.png"
else
    # Create placeholder icon
    echo "Warning: No icon found, creating placeholder"
    # You can use ImageMagick to create a placeholder:
    # convert -size 256x256 xc:blue -pointsize 60 -draw "text 50,150 'SV'" "$APPDIR/smartvoice.png"
fi

echo "AppDir structure created!"
echo

echo "Step 4: Building AppImage..."
echo "----------------------------"

# Check if appimagetool is available
if ! command -v appimagetool &> /dev/null; then
    echo "ERROR: appimagetool not found!"
    echo "Please download from: https://github.com/AppImage/AppImageKit/releases"
    echo "Or install with: wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    echo "                 chmod +x appimagetool-x86_64.AppImage"
    echo "                 sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool"
    exit 1
fi

# Build AppImage
OUTPUT_FILE="$PROJECT_ROOT/SmartVoice-${VERSION}-x86_64.AppImage"
appimagetool "$APPDIR" "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo
    echo "============================================"
    echo "Build completed successfully!"
    echo "============================================"
    echo
    echo "AppImage created: $(basename "$OUTPUT_FILE")"
    echo "Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo
    echo "You can now distribute this AppImage file."
    echo
    echo "To run: chmod +x $OUTPUT_FILE && $OUTPUT_FILE"
    echo
else
    echo "ERROR: AppImage creation failed!"
    exit 1
fi
