@echo off
REM Build Windows installer for WhisperX SmartVoice
REM Prerequisites:
REM   - Python 3.9+ with PyInstaller installed
REM   - NSIS (Nullsoft Scriptable Install System) installed
REM   - UV package manager (optional, for dependency management)

echo ============================================
echo WhisperX SmartVoice Windows Installer Build
echo ============================================
echo.

REM Change to project root
cd /d "%~dp0..\..\..\"

echo Step 1: Building Launcher executable...
echo ----------------------------------------
python -m PyInstaller launcher.spec --clean
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Launcher build failed!
    pause
    exit /b 1
)
echo Launcher built successfully!
echo.

echo Step 2: Building SmartVoice main executable...
echo ----------------------------------------------
python -m PyInstaller smartvoice.spec --clean
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: SmartVoice build failed!
    pause
    exit /b 1
)
echo SmartVoice built successfully!
echo.

echo Step 3: Creating Windows installer with NSIS...
echo -----------------------------------------------
REM Check if NSIS is installed
where makensis >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NSIS (makensis) not found in PATH!
    echo Please install NSIS from https://nsis.sourceforge.io/
    pause
    exit /b 1
)

makensis scripts\installers\windows\installer.nsi
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Installer creation failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Build completed successfully!
echo ============================================
echo.
echo Installer created: SmartVoice-Setup-3.4.2.exe
echo.
echo You can now distribute this installer file.
echo.

pause
