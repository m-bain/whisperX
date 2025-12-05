# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperX Launcher

Builds a standalone launcher executable that:
- Includes Python interpreter
- Manages dependencies (CPU/GPU)
- Checks for updates
- Launches main SmartVoice application

Build command:
    pyinstaller launcher.spec
"""

block_cipher = None

# Analysis: Collect all launcher files and dependencies
a = Analysis(
    ['whisperx/launcher/launcher_main.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=[
        # Include launcher core modules
        ('whisperx/launcher/core', 'whisperx/launcher/core'),
        ('whisperx/launcher/__init__.py', 'whisperx/launcher'),
        # Include version file
        ('whisperx/__version__.py', 'whisperx'),
        # Include icon if present
        # ('whisperx/launcher/resources/icon.ico', 'resources'),
    ],
    hiddenimports=[
        'whisperx.launcher.core',
        'whisperx.launcher.core.hardware_detection',
        'whisperx.launcher.core.version_manager',
        'whisperx.launcher.core.dependency_manager',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'requests',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
        'subprocess',
        'json',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy dependencies (will be installed by launcher)
        'torch',
        'torchaudio',
        'torchvision',
        'whisperx.transcribe',
        'whisperx.alignment',
        'whisperx.diarize',
        'faster_whisper',
        'transformers',
        'pyannote',
        'matplotlib',
        'scipy',
        'sklearn',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SmartVoiceLauncher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application (no console window)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='whisperx/launcher/resources/icon.ico' if os.path.exists('whisperx/launcher/resources/icon.ico') else None,
)

# Collect all files into a directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartVoiceLauncher'
)
