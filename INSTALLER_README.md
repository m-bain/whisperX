# WhisperX SmartVoice Installer System

## Overview

WhisperX SmartVoice uses a **hybrid launcher architecture** that separates the lightweight launcher from the heavy AI dependencies. This provides:

- **Small initial download** (~100-150 MB installer)
- **Smart dependency management** (CPU vs GPU auto-detection)
- **Easy updates** (no need to re-download entire PyTorch)
- **Version management** and seamless reinstallation
- **User data preservation** across updates

---

## Architecture

### Components

```
SmartVoice Installation
├── Launcher (50-100 MB)
│   ├── Hardware detection
│   ├── Dependency manager
│   ├── Version checker
│   └── Update system
│
├── Main Application (50-100 MB)
│   ├── SmartVoice GUI
│   ├── WhisperX core
│   └── UI assets
│
└── Dependencies (Managed by Launcher)
    ├── PyTorch (CPU: ~200 MB | CUDA: ~2 GB)
    ├── WhisperX deps (~500 MB)
    └── Models (downloaded on-demand)
```

### User Data (Preserved)

```
~/.whisperx_app/
├── config.json              # User preferences
├── presets.json             # Saved presets
├── transcription_history.db # All transcription records
├── dependency_config.json   # Installed dependencies info
├── version_cache.json       # Update check cache
└── installed.lock           # Installation marker
```

---

## Building Installers

### Prerequisites

**All Platforms:**
- Python 3.9 - 3.12
- PyInstaller: `pip install pyinstaller`
- UV package manager (optional): `pip install uv`

**Windows:**
- NSIS (Nullsoft Scriptable Install System)
  - Download: https://nsis.sourceforge.io/Download
  - Add to PATH during installation

**Linux:**
- appimagetool
  ```bash
  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x appimagetool-x86_64.AppImage
  sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
  ```

---

### Build Instructions

#### Windows

```cmd
# Navigate to installer directory
cd scripts\installers\windows

# Run build script
build_installer.bat

# Output: SmartVoice-Setup-3.4.2.exe (~100 MB)
```

**Manual Steps:**
1. Build executables:
   ```cmd
   pyinstaller launcher.spec --clean
   pyinstaller smartvoice.spec --clean
   ```

2. Create installer:
   ```cmd
   makensis scripts\installers\windows\installer.nsi
   ```

#### Linux

```bash
# Navigate to installer directory
cd scripts/installers/linux

# Run build script
./build_appimage.sh

# Output: SmartVoice-3.4.2-x86_64.AppImage (~100 MB)
```

**Manual Steps:**
1. Build executables:
   ```bash
   python3 -m PyInstaller launcher.spec --clean
   python3 -m PyInstaller smartvoice.spec --clean
   ```

2. Create AppImage:
   ```bash
   # Follow structure in build_appimage.sh
   appimagetool SmartVoice.AppDir SmartVoice-3.4.2-x86_64.AppImage
   ```

---

## Installation Flow

### First-Time Installation

1. **User downloads installer** (~100 MB)
   - Windows: `SmartVoice-Setup-3.4.2.exe`
   - Linux: `SmartVoice-3.4.2-x86_64.AppImage`

2. **Installer extracts files**
   - Launcher application
   - Main SmartVoice application
   - Creates shortcuts

3. **User launches SmartVoice** (via shortcut)
   - Launcher opens automatically

4. **First-time setup** (automatic)
   - Detects hardware (CPU/NVIDIA GPU/AMD GPU)
   - Shows recommended installation type
   - User can choose:
     - **CPU Only** (recommended, ~700 MB download)
     - **CUDA 11.8** (for older GPUs, ~2.5 GB)
     - **CUDA 12.1** (for newer GPUs, ~2.5 GB)

5. **Downloads dependencies**
   - Shows progress bar
   - Downloads and installs PyTorch
   - Installs WhisperX dependencies
   - Verifies installation

6. **Ready to use!**
   - "Launch SmartVoice" button enabled
   - Models downloaded on-demand during first transcription

---

### Subsequent Launches

1. User launches SmartVoice
2. Launcher checks if dependencies installed
3. If yes → Launches main application immediately
4. If no → Shows setup wizard again

---

## Update System

### Checking for Updates

The launcher automatically checks for updates:
- On startup (max once per 6 hours, cached)
- Via "Check for Updates" button in Updates tab

### Update Process

#### Small Updates (Patch versions: 3.4.2 → 3.4.3)
- User clicks "Download and Install Update"
- Downloads new installer (~100 MB)
- Runs installer
- Installer replaces application files
- **Dependencies not touched** (no re-download)
- User data preserved automatically

#### Major Updates (Minor/Major versions: 3.4.x → 3.5.0)
- Similar to small updates, but:
- May require dependency reinstallation
- Launcher detects and prompts user
- Re-runs first-time setup for new dependencies

---

## Switching Between CPU and GPU

Users can switch installation types without reinstalling the entire application:

1. Open Launcher
2. Go to "Setup" tab
3. Click "Switch to GPU" (or "Switch to CPU")
4. Confirms action
5. Launcher:
   - Uninstalls current PyTorch
   - Downloads and installs new PyTorch version
   - Verifies installation
6. Done! (~5-10 minutes, 2 GB download for GPU)

---

## Reinstallation

To completely reinstall (preserving user data):

### Option 1: Via Launcher
1. Open Launcher
2. Go to "Setup" tab
3. Click "Reinstall"
4. Confirms action
5. Launcher clears installation markers
6. Re-runs first-time setup

### Option 2: Via Installer
1. Run installer again
2. Choose same installation directory
3. Installer detects previous installation
4. Overwrites application files
5. **User data in `~/.whisperx_app/` preserved automatically**

---

## Uninstallation

### Windows
1. Go to "Add or Remove Programs"
2. Find "WhisperX SmartVoice"
3. Click "Uninstall"
4. Uninstaller removes:
   - Application files
   - Shortcuts
   - Registry entries
5. **User data preserved** (manual deletion if desired)

### Linux (AppImage)
1. Simply delete the `.AppImage` file
2. Optionally delete `~/.whisperx_app/` for complete removal

---

## Launcher Features

### Setup Tab
- **System Information**: Shows detected hardware
- **Installation Status**: Current installation type and versions
- **Install Dependencies**: First-time setup
- **Switch CPU/GPU**: Change PyTorch version without reinstalling app
- **Reinstall**: Clear and reinstall dependencies
- **Installation Log**: Real-time progress and debug info

### Updates Tab
- **Current Version**: Shows installed version
- **Check for Updates**: Manual update check
- **Download Update**: Download and install new version
- **Release Notes**: View changelog for new version

### Settings Tab
- **Auto-check Updates**: Toggle automatic update checking
- **Data Location**: Shows where user data is stored
- **About**: Version and credits

---

## File Structure After Installation

### Windows
```
C:\Program Files\WhisperX\SmartVoice\
├── Launcher\
│   ├── SmartVoiceLauncher.exe
│   ├── ... (launcher files)
│
├── SmartVoice\
│   ├── SmartVoice.exe
│   ├── ... (main app files)
│
└── Uninstall.exe

%USERPROFILE%\.whisperx_app\
├── config.json
├── presets.json
├── transcription_history.db
├── dependency_config.json
├── version_cache.json
└── installed.lock
```

### Linux
```
(AppImage is self-contained, no installation directory)

~/.whisperx_app/
├── config.json
├── presets.json
├── transcription_history.db
├── dependency_config.json
├── version_cache.json
└── installed.lock
```

---

## Development

### Project Structure

```
whisperX/
├── whisperx/
│   ├── __version__.py                    # Single source of truth for version
│   ├── launcher/
│   │   ├── launcher_main.py              # Main launcher application
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── hardware_detection.py    # GPU/CUDA detection
│   │   │   ├── version_manager.py       # Update checking
│   │   │   └── dependency_manager.py    # Dependency installation
│   │   └── ui/                           # (future) UI resources
│   │
│   └── appSmartVoice/                    # Main application
│       └── main.py                       # SmartVoice GUI entry point
│
├── scripts/
│   └── installers/
│       ├── windows/
│       │   ├── installer.nsi             # NSIS script
│       │   └── build_installer.bat       # Windows build script
│       └── linux/
│           └── build_appimage.sh         # Linux build script
│
├── launcher.spec                         # PyInstaller spec for launcher
├── smartvoice.spec                       # PyInstaller spec for main app
└── INSTALLER_README.md                   # This file
```

### Testing the Launcher

Run the launcher in development mode:

```bash
# Navigate to project root
cd whisperX

# Run launcher directly
python whisperx/launcher/launcher_main.py

# Or with logging
python -m whisperx.launcher.launcher_main
```

### Modifying the Installer

1. Update version in `whisperx/__version__.py`
2. Update version in installer scripts:
   - `scripts/installers/windows/installer.nsi` (line 12)
   - `scripts/installers/linux/build_appimage.sh` (line 19)
3. Rebuild installers

---

## Troubleshooting

### Launcher won't start after installation

**Windows:**
- Check if antivirus blocked the executable
- Run as administrator
- Check Windows Event Viewer for errors

**Linux:**
- Ensure AppImage is executable: `chmod +x SmartVoice-*.AppImage`
- Check if FUSE is installed: `sudo apt install fuse libfuse2`
- Try extracting and running: `./SmartVoice-*.AppImage --appimage-extract && ./squashfs-root/AppRun`

### Dependencies installation fails

1. Check internet connection
2. Check disk space (need ~5-10 GB free)
3. View installation log in launcher for specific error
4. Try manual installation:
   ```bash
   # In launcher log, find Python executable path
   /path/to/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

### CUDA not detected despite having NVIDIA GPU

1. Install/update NVIDIA drivers
2. Check CUDA installation: `nvidia-smi`
3. Restart computer
4. In launcher, manually select CUDA version in dropdown

### Update check fails

1. Check internet connection
2. Check if GitHub API is accessible: `curl https://api.github.com/repos/xlazarik/whisperX/releases/latest`
3. Clear cache: Delete `~/.whisperx_app/version_cache.json`
4. Try manual update check in launcher

### Application won't launch from launcher

1. Check if dependencies are installed (Setup tab)
2. Try reinstalling dependencies
3. Check launcher log for errors
4. Verify Python environment in dependency_config.json

---

## FAQ

### Q: Where is my transcription history stored?
**A:** In `~/.whisperx_app/transcription_history.db` (SQLite database). This is preserved across updates and reinstalls.

### Q: How much disk space do I need?
**A:**
- Installer: ~100 MB
- Dependencies (CPU): ~1 GB
- Dependencies (GPU): ~3-4 GB
- Models (downloaded on-demand): ~1-3 GB per model
- **Total: 2-8 GB depending on configuration**

### Q: Can I have both CPU and GPU versions?
**A:** The launcher manages only one active installation at a time. Use "Switch" to change between them.

### Q: Do I need to reinstall when switching between CPU and GPU?
**A:** No! The launcher only reinstalls PyTorch (~5-10 minutes), not the entire application.

### Q: Will my data be lost if I uninstall?
**A:** No, user data in `~/.whisperx_app/` is preserved. You can manually delete it if desired.

### Q: Can I install on multiple machines with the same configuration?
**A:** Yes, copy your `~/.whisperx_app/config.json` and `presets.json` to the new machine after installation.

### Q: How do I report bugs?
**A:** Please open an issue on GitHub: https://github.com/xlazarik/whisperX/issues

---

## License

WhisperX is licensed under the BSD-2-Clause License. See LICENSE file for details.

---

## Credits

- **WhisperX Core**: Max Bain (https://github.com/m-bain/whisperX)
- **Launcher System**: Created for this fork
- **Qt Framework**: PySide6
- **Custom Widgets**: QT-PyQt-PySide-Custom-Widgets

---

## Version History

### v3.4.2 (Current)
- Initial launcher system implementation
- CPU/GPU switching support
- Version management and updates
- User data preservation

---

For more information, visit: https://github.com/xlazarik/whisperX
