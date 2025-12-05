# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for WhisperX SmartVoice Main Application

Builds the main SmartVoice GUI application (WITHOUT PyTorch/heavy dependencies).
Dependencies are managed by the launcher and installed separately.

Build command:
    pyinstaller smartvoice.spec
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Helper function to collect Custom_Widgets
def collect_custom_widgets():
    """Collect all Custom_Widgets package files."""
    try:
        import Custom_Widgets
        pkg_path = Path(Custom_Widgets.__file__).parent

        # Collect the entire Custom_Widgets package
        return [(str(pkg_path), 'Custom_Widgets')]
    except ImportError:
        print("WARNING: Custom_Widgets not found in environment!")
        return []

# Helper to check if module exists before adding to hiddenimports
def get_safe_hiddenimports():
    """Get list of hidden imports, only including modules that exist."""
    safe_imports = [
        # Qt (always present)
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtSql',
    ]

    # Custom Widgets - comprehensive list
    custom_widgets_imports = [
        'Custom_Widgets',
        'Custom_Widgets.QAppSettings',
        'Custom_Widgets.QCustomTipOverlay',
        'Custom_Widgets.QCustomLoadingIndicators',
        'Custom_Widgets.QCustomQStackedWidget',
        'Custom_Widgets.QCustomSlideMenu',
        'Custom_Widgets.QCustomQPushButtonGroup',
        'Custom_Widgets.Qss',
        'Custom_Widgets.AnalogGaugeWidget',
        'Custom_Widgets.QCustomModals',
    ]

    # Try to import Custom_Widgets modules
    for module in custom_widgets_imports:
        try:
            __import__(module)
            safe_imports.append(module)
        except ImportError:
            print(f"Note: {module} not found, skipping")

    # WhisperX app modules (should exist)
    whisperx_app_imports = [
        'whisperx.app.transcription_manager',
        'whisperx.app.transcription_workers',
        'whisperx.app.history_manager',
        'whisperx.app.app_config',
        'whisperx.SubtitlesProcessor',
    ]

    for module in whisperx_app_imports:
        try:
            __import__(module)
            safe_imports.append(module)
        except ImportError:
            print(f"Note: {module} not found, will be collected via datas")

    # Other common imports
    safe_imports.extend([
        'numpy',
        'pandas',
        'psutil',
        'sqlite3',
        'json',
        'pathlib',
    ])

    return safe_imports

# Get Custom_Widgets data
custom_widgets_data = collect_custom_widgets()

# Analysis: Collect main application files
a = Analysis(
    ['whisperx/appSmartVoice/main.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=[
        # UI files and resources
        ('whisperx/appSmartVoice/ui', 'whisperx/appSmartVoice/ui'),
        ('whisperx/appSmartVoice/json-styles', 'whisperx/appSmartVoice/json-styles'),
        ('whisperx/appSmartVoice/Qss', 'whisperx/appSmartVoice/Qss'),
        ('whisperx/appSmartVoice/generated-files', 'whisperx/appSmartVoice/generated-files'),
        ('whisperx/appSmartVoice/src', 'whisperx/appSmartVoice/src'),

        # Include whisperx core modules
        ('whisperx/__init__.py', 'whisperx'),
        ('whisperx/__version__.py', 'whisperx'),
        ('whisperx/app/*.py', 'whisperx/app'),
        ('whisperx/*.py', 'whisperx'),

        # Include Custom_Widgets package
        *custom_widgets_data,
    ],
    hiddenimports=get_safe_hiddenimports(),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML dependencies - they'll be loaded from installed environment
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
