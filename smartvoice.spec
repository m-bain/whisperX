# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperX SmartVoice Main Application

Builds the main SmartVoice GUI application (WITHOUT PyTorch/heavy dependencies).
Dependencies are managed by the launcher and installed separately.

Build command:
    pyinstaller smartvoice.spec
"""

import os
from pathlib import Path

block_cipher = None

# Analysis: Collect main application files
a = Analysis(
    ['whisperx/appSmartVoice/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # UI files and resources
        ('whisperx/appSmartVoice/ui', 'whisperx/appSmartVoice/ui'),
        ('whisperx/appSmartVoice/json-styles', 'whisperx/appSmartVoice/json-styles'),
        ('whisperx/appSmartVoice/Qss', 'whisperx/appSmartVoice/Qss'),
        ('whisperx/appSmartVoice/generated-files', 'whisperx/appSmartVoice/generated-files'),

        # Include whisperx core modules (they import lazily)
        ('whisperx/*.py', 'whisperx'),
        ('whisperx/app', 'whisperx/app'),
        ('whisperx/vads', 'whisperx/vads'),

        # Version file
        ('whisperx/__version__.py', 'whisperx'),
    ],
    hiddenimports=[
        # Qt and Custom Widgets
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'Custom_Widgets',
        'Custom_Widgets.QAppSettings',
        'Custom_Widgets.QCustomTipOverlay',
        'Custom_Widgets.QCustomLoadingIndicators',

        # WhisperX core (lazy loaded)
        'whisperx',
        'whisperx.transcribe',
        'whisperx.alignment',
        'whisperx.diarize',
        'whisperx.asr',
        'whisperx.audio',
        'whisperx.SubtitlesProcessor',
        'whisperx.app.transcription_manager',
        'whisperx.app.transcription_workers',
        'whisperx.app.history_manager',
        'whisperx.app.app_config',

        # AI/ML frameworks (will use installed versions)
        'torch',
        'torchaudio',
        'faster_whisper',
        'transformers',
        'pyannote.audio',
        'ctranslate2',

        # Other dependencies
        'numpy',
        'pandas',
        'nltk',
        'onnxruntime',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='SmartVoice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='whisperx/appSmartVoice/resources/icon.ico' if os.path.exists('whisperx/appSmartVoice/resources/icon.ico') else None,
)

# Collect all files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartVoice'
)
