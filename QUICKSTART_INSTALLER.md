# Quick Start: Building WhisperX SmartVoice Installers

This guide will help you build the SmartVoice installer in **5 minutes**.

---

## Prerequisites

Install these first:

### 1. Python Requirements
```bash
pip install pyinstaller packaging
```

### 2. Platform-Specific Tools

**Windows:**
- Download NSIS: https://nsis.sourceforge.io/Download
- Install and make sure `makensis` is in your PATH

**Linux:**
```bash
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage
sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
```

---

## Build Installer (Windows)

```cmd
cd whisperX

# Run automated build script
scripts\installers\windows\build_installer.bat
```

**Output:** `SmartVoice-Setup-3.4.2.exe` (~100 MB)

---

## Build Installer (Linux)

```bash
cd whisperX

# Run automated build script
chmod +x scripts/installers/linux/build_appimage.sh
./scripts/installers/linux/build_appimage.sh
```

**Output:** `SmartVoice-3.4.2-x86_64.AppImage` (~100 MB)

---

## Test the Installer

### Windows
1. Run `SmartVoice-Setup-3.4.2.exe`
2. Follow installation wizard
3. Launch "SmartVoice Launcher" from Start Menu
4. Click "Install Dependencies" (choose CPU or GPU)
5. Wait for installation (~5-10 minutes)
6. Click "Launch SmartVoice"

### Linux
1. Make executable: `chmod +x SmartVoice-3.4.2-x86_64.AppImage`
2. Run: `./SmartVoice-3.4.2-x86_64.AppImage`
3. Launcher opens automatically
4. Click "Install Dependencies" (choose CPU or GPU)
5. Wait for installation (~5-10 minutes)
6. Click "Launch SmartVoice"

---

## What Happens During Installation?

1. **Installer extracts** launcher and main app (~100 MB)
2. **Launcher detects** your hardware (CPU/GPU)
3. **Downloads PyTorch** appropriate for your system:
   - CPU: ~700 MB
   - GPU (CUDA): ~2.5 GB
4. **Installs WhisperX** and dependencies (~500 MB)
5. **Ready to use!** Models download on first transcription

---

## Distribution

Once built, distribute the installer file:

- **Windows**: `SmartVoice-Setup-3.4.2.exe`
- **Linux**: `SmartVoice-3.4.2-x86_64.AppImage`

Users just need to:
1. Download the installer
2. Run it
3. Follow first-time setup
4. Start transcribing!

---

## Troubleshooting

**Build fails with "PyInstaller not found":**
```bash
pip install pyinstaller
```

**Windows: "makensis not found":**
- Install NSIS and add to PATH
- Or manually run: `"C:\Program Files (x86)\NSIS\makensis.exe" scripts\installers\windows\installer.nsi`

**Linux: "appimagetool not found":**
```bash
# Re-download and install appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage
sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
```

**Hidden imports missing:**
- Edit `launcher.spec` or `smartvoice.spec`
- Add missing modules to `hiddenimports` list
- Rebuild

---

## Next Steps

- Read full documentation: `INSTALLER_README.md`
- Test on different machines
- Set up automated builds with GitHub Actions
- Create release on GitHub with installers attached

---

For detailed information, see [INSTALLER_README.md](INSTALLER_README.md)
