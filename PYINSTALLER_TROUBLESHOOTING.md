# PyInstaller Build Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: Custom_Widgets Module Not Found

**Symptoms:**
```
ModuleNotFoundError: No module named 'Custom_Widgets'
```

**Solution:**
1. Ensure Custom_Widgets is installed in your build environment:
   ```bash
   pip install QT-PyQt-PySide-Custom-Widgets
   ```

2. Verify installation:
   ```python
   python -c "import Custom_Widgets; print(Custom_Widgets.__file__)"
   ```

3. Rebuild with the updated spec file (which now includes automatic Custom_Widgets collection)

---

### Issue 2: Hidden Import Errors During Build

**Symptoms:**
```
ERROR: Hidden import 'whisperx.transcribe' not found
ERROR: Hidden import 'whisperx.alignment' not found
```

**Why This Happens:**
- These are **warnings, not fatal errors**
- WhisperX modules use lazy loading
- They're not needed at build time, only at runtime
- They'll be loaded from the environment installed by the launcher

**Solution:**
- **Ignore these warnings** - they're expected
- The spec file now handles this gracefully with `get_safe_hiddenimports()`
- If build completes, test the executable

---

### Issue 3: Build Succeeds But Executable Crashes

**Symptoms:**
- Build completes without errors
- Running the .exe crashes immediately
- No window appears

**Debugging Steps:**

1. **Run with console output** (Windows):
   Edit spec file, change `console=False` to `console=True`:
   ```python
   exe = EXE(
       ...
       console=True,  # Change this temporarily
       ...
   )
   ```
   Rebuild and run to see error messages.

2. **Check for missing DLLs** (Windows):
   ```bash
   # Use Dependency Walker or similar tool
   depends.exe dist/SmartVoice/SmartVoice.exe
   ```

3. **Verify Python can import modules**:
   ```bash
   cd dist/SmartVoice
   ./SmartVoice.exe  # Should show error in console
   ```

---

### Issue 4: Import Errors at Runtime

**Symptoms:**
```
ImportError: cannot import name 'XXX' from 'YYY'
```

**Solution:**
Add missing module to `hiddenimports` in spec file:
```python
hiddenimports=[
    ...
    'your.missing.module',
]
```

---

### Issue 5: UI Files Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'whisperx/appSmartVoice/ui/interface.ui'
```

**Solution:**
Ensure all UI resources are in the `datas` section:
```python
datas=[
    ('whisperx/appSmartVoice/ui', 'whisperx/appSmartVoice/ui'),
    ('whisperx/appSmartVoice/json-styles', 'whisperx/appSmartVoice/json-styles'),
    ('whisperx/appSmartVoice/Qss', 'whisperx/appSmartVoice/Qss'),
]
```

---

## Testing Checklist

After building, test these scenarios:

### Launcher Tests
- [ ] Launcher executable starts without errors
- [ ] Window appears with correct title
- [ ] System information displays correctly
- [ ] All tabs are accessible (Setup, Updates, Settings)
- [ ] Hardware detection works (shows CPU/GPU info)
- [ ] No import errors in console output

### Main App Tests
- [ ] SmartVoice executable starts (when launched by launcher)
- [ ] Main window appears
- [ ] Left menu navigation works
- [ ] Can select audio file
- [ ] No Custom_Widgets errors
- [ ] UI styling loads correctly (colors, fonts)

---

## Build Environment Requirements

### Python Environment
```bash
# Create clean environment
python -m venv build_env
source build_env/bin/activate  # Linux
# or
build_env\Scripts\activate  # Windows

# Install dependencies
pip install pyinstaller
pip install PySide6
pip install QT-PyQt-PySide-Custom-Widgets
pip install requests packaging

# Install whisperx in development mode
pip install -e .
```

### Windows-Specific
- **MSVC Redistributable**: Required for PySide6
  - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- **NSIS**: For creating installer
  - Download: https://nsis.sourceforge.io/Download

### Linux-Specific
- **System libraries**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y \
      libxcb-xinerama0 \
      libxcb-cursor0 \
      libxkbcommon-x11-0 \
      libxcb-icccm4 \
      libxcb-image0 \
      libxcb-keysyms1 \
      libxcb-randr0 \
      libxcb-render-util0 \
      libxcb-shape0
  ```

---

## Advanced: Creating Custom Hook

If a module still won't include properly, create a custom PyInstaller hook:

1. Create `hooks/hook-Custom_Widgets.py`:
```python
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('Custom_Widgets')
```

2. Update spec file:
```python
hookspath=['hooks'],  # Add hooks directory
```

---

## Clean Rebuild

If build is corrupted or has issues:

```bash
# Remove build artifacts
rm -rf build/ dist/ *.spec

# Or on Windows:
rmdir /s build dist
del *.spec

# Rebuild from scratch
pyinstaller smartvoice.spec --clean
```

---

## Debugging Tips

### 1. Enable Debug Mode
Edit spec file:
```python
exe = EXE(
    ...
    debug=True,  # Enable debug output
    console=True,  # Show console
)
```

### 2. Check What's Included
```bash
# List all files in build
cd dist/SmartVoice
find . -type f  # Linux
dir /s  # Windows
```

### 3. Test Import in Python
```python
# Test if modules can be imported
import sys
sys.path.insert(0, 'dist/SmartVoice')

import Custom_Widgets
import whisperx.app.transcription_manager
print("Imports successful!")
```

### 4. Use PyInstaller's --log-level
```bash
pyinstaller smartvoice.spec --log-level=DEBUG
```

---

## Known Limitations

1. **Heavy ML Dependencies**
   - PyTorch, transformers, etc. are NOT included in executable
   - They're installed by the launcher on first run
   - This is by design to keep installer small

2. **Platform-Specific Builds**
   - Windows build only works on Windows
   - Linux build only works on Linux
   - Cross-compilation not supported

3. **Python Version**
   - Build must use Python 3.9-3.12
   - Match the version your app requires

---

## Getting Help

If you're still stuck:

1. **Check PyInstaller logs** in `build/` directory
2. **Search PyInstaller issues**: https://github.com/pyinstaller/pyinstaller/issues
3. **Check Custom_Widgets compatibility**: https://github.com/KhamisiKibet/QT-PyQt-PySide-Custom-Widgets
4. **Test without PyInstaller** first:
   ```bash
   python whisperx/appSmartVoice/main.py
   ```
   If this fails, it's not a PyInstaller issue.

---

## Success Indicators

Your build is successful when:
- ✓ Build completes without fatal errors
- ✓ `dist/SmartVoice/SmartVoice.exe` exists
- ✓ Running executable shows the launcher window
- ✓ No "module not found" errors
- ✓ UI elements load and display correctly
- ✓ Executable size is reasonable (~50-150 MB)

---

## Quick Fix Summary

**For the specific issues you encountered:**

1. **Custom_Widgets not found** → Fixed by:
   - Added `collect_custom_widgets()` function in spec
   - Collects entire Custom_Widgets package automatically
   - Added comprehensive hiddenimports list

2. **WhisperX import errors** → Fixed by:
   - Changed to use `get_safe_hiddenimports()`
   - Only adds imports that actually exist
   - Errors are now informational, not fatal

**Rebuild with:**
```bash
pyinstaller smartvoice.spec --clean
```

The updated spec file should now handle these issues automatically.
