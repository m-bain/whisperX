@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0bin;%PATH%"
set "HF_HOME=%~dp0models\huggingface"
set "TORCH_HOME=%~dp0models\torch"
set "XDG_CACHE_HOME=%~dp0models\cache"
if not exist "%~dp0output" mkdir "%~dp0output"
if "%~1"=="" (
  echo Drag an audio file onto run.bat, or paste its path below.
  set /p "AUDIO=Audio path: "
  if not defined AUDIO exit /b 1
  "%~dp0python\python.exe" -m whisperx "%AUDIO%" --output_dir "%~dp0output"
) else (
  "%~dp0python\python.exe" -m whisperx %* --output_dir "%~dp0output"
)
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo WhisperX exited with code %RC%.
pause
exit /b %RC%
