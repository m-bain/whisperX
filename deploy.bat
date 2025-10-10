@echo off
setlocal enabledelayedexpansion

:: WhisperX Docker Deployment Script for Windows
:: This script helps you easily deploy WhisperX with Docker on Windows

title WhisperX Docker Deployment

:: Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo [SUCCESS] Docker and Docker Compose are installed

:: Setup directories
echo [INFO] Setting up directories...
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "models" mkdir models
echo [SUCCESS] Directories created: input/, output/, models/

:: Setup environment file
if not exist ".env" (
    echo [INFO] Creating .env file from template...
    copy ".env.template" ".env" >nul
    echo [WARNING] Please edit .env file to add your Hugging Face token for speaker diarization
    echo [INFO] Get your token from: https://huggingface.co/settings/tokens
) else (
    echo [INFO] .env file already exists
)

:: Check for GPU support
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] NVIDIA GPU detected
    set GPU_AVAILABLE=true
) else (
    echo [WARNING] No NVIDIA GPU detected or nvidia-smi not available
    set GPU_AVAILABLE=false
)

:menu
cls
echo ===================================
echo     WhisperX Docker Deployment
echo ===================================
echo 1. Deploy with GPU (Recommended)
echo 2. Deploy with CPU only
echo 3. Deploy Web API with GPU
echo 4. Deploy Web API with CPU
echo 5. Stop all containers
echo 6. View logs
echo 7. Transcribe audio file
echo 8. Clean up everything
echo 9. Exit
echo ===================================

set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto deploy_gpu
if "%choice%"=="2" goto deploy_cpu
if "%choice%"=="3" goto deploy_api_gpu
if "%choice%"=="4" goto deploy_api_cpu
if "%choice%"=="5" goto stop_containers
if "%choice%"=="6" goto view_logs
if "%choice%"=="7" goto transcribe_audio
if "%choice%"=="8" goto cleanup
if "%choice%"=="9" goto exit
echo [ERROR] Invalid choice. Please try again.
pause
goto menu

:deploy_gpu
if "%GPU_AVAILABLE%"=="false" (
    echo [ERROR] GPU not available. Please use CPU mode.
    pause
    goto menu
)
echo [INFO] Deploying WhisperX with GPU support...
docker-compose --profile gpu up -d
echo [SUCCESS] WhisperX GPU container is running!
echo [INFO] Use: docker exec whisperx-gpu python -m whisperx /input/audio.wav --output_dir /output
pause
goto menu

:deploy_cpu
echo [INFO] Deploying WhisperX with CPU only...
docker-compose --profile cpu up -d
echo [SUCCESS] WhisperX CPU container is running!
echo [INFO] Use: docker exec whisperx-cpu python -m whisperx /input/audio.wav --output_dir /output --device cpu --compute_type int8
pause
goto menu

:deploy_api_gpu
if "%GPU_AVAILABLE%"=="false" (
    echo [ERROR] GPU not available. Please use CPU mode.
    pause
    goto menu
)
echo [INFO] Deploying WhisperX Web API with GPU support...
docker-compose --profile api-gpu up -d
echo [SUCCESS] WhisperX Web API (GPU) is running!
echo [INFO] Access the web interface at: http://localhost:8000
start http://localhost:8000
pause
goto menu

:deploy_api_cpu
echo [INFO] Deploying WhisperX Web API with CPU only...
docker-compose --profile api-cpu up -d
echo [SUCCESS] WhisperX Web API (CPU) is running!
echo [INFO] Access the web interface at: http://localhost:8000
start http://localhost:8000
pause
goto menu

:stop_containers
echo [INFO] Stopping all WhisperX containers...
docker-compose down
echo [SUCCESS] All containers stopped
pause
goto menu

:view_logs
echo Select container to view logs:
echo 1. GPU container
echo 2. CPU container
echo 3. API GPU container
echo 4. API CPU container
set /p log_choice="Enter choice (1-4): "

if "%log_choice%"=="1" docker-compose logs -f whisperx-gpu
if "%log_choice%"=="2" docker-compose logs -f whisperx-cpu
if "%log_choice%"=="3" docker-compose logs -f whisperx-api-gpu
if "%log_choice%"=="4" docker-compose logs -f whisperx-api-cpu
if not "%log_choice%"=="1" if not "%log_choice%"=="2" if not "%log_choice%"=="3" if not "%log_choice%"=="4" echo [ERROR] Invalid choice
pause
goto menu

:transcribe_audio
echo [INFO] Audio files in input/ directory:
dir input\*.wav input\*.mp3 input\*.m4a input\*.flac input\*.ogg /b 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] No audio files found in input/ directory
    echo [INFO] Please place audio files (.wav, .mp3, .m4a, .flac, .ogg) in the input/ directory
    pause
    goto menu
)

echo.
set /p audio_file="Enter audio filename (from input/ directory): "

if not exist "input\%audio_file%" (
    echo [ERROR] File input\%audio_file% not found
    pause
    goto menu
)

echo Select transcription mode:
echo 1. GPU (fast)
echo 2. CPU (slower)
set /p mode_choice="Enter choice (1-2): "

echo Additional options:
set /p diarize="Enable speaker diarization? (y/n): "
set /p model="Model (tiny/base/small/medium/large-v2) [default: large-v2]: "

if "%model%"=="" set model=large-v2

set cmd=python -m whisperx /input/%audio_file% --output_dir /output --model %model%

if /i "%diarize%"=="y" set cmd=%cmd% --diarize

if "%mode_choice%"=="1" (
    set container=whisperx-gpu
) else if "%mode_choice%"=="2" (
    set container=whisperx-cpu
    set cmd=%cmd% --device cpu --compute_type int8
) else (
    echo [ERROR] Invalid choice
    pause
    goto menu
)

echo [INFO] Running transcription...
echo [INFO] Command: docker exec %container% %cmd%

docker exec %container% %cmd%
if %errorlevel% equ 0 (
    echo [SUCCESS] Transcription completed! Check output/ directory for results.
    start output
) else (
    echo [ERROR] Transcription failed. Check container logs for details.
)
pause
goto menu

:cleanup
echo [WARNING] This will remove all containers, images, and cached models!
set /p confirm="Are you sure? (y/N): "

if /i "%confirm%"=="y" (
    echo [INFO] Stopping containers...
    docker-compose down
    
    echo [INFO] Removing images...
    for /f "tokens=3" %%i in ('docker images ^| findstr whisperx') do docker rmi %%i 2>nul
    
    echo [INFO] Cleaning up model cache...
    del /q models\*.* 2>nul
    
    echo [SUCCESS] Cleanup completed
) else (
    echo [INFO] Cleanup cancelled
)
pause
goto menu

:exit
echo [INFO] Goodbye!
pause
exit /b 0