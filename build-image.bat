@echo off
setlocal

REM WhisperX Docker Image Builder for Windows
REM This script builds and tags Docker images for distribution

echo 🐳 Building WhisperX Enhanced Docker Images
echo ============================================

REM Configuration
set DOCKER_USERNAME=yourusername
set IMAGE_NAME=whisperx-enhanced
set VERSION=latest

REM Build GPU version
echo 📦 Building GPU version...
docker build -t %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION% .
if %errorlevel% neq 0 exit /b %errorlevel%

docker build -t %DOCKER_USERNAME%/%IMAGE_NAME%:gpu .
if %errorlevel% neq 0 exit /b %errorlevel%

REM Build CPU version
echo 📦 Building CPU version...
docker build -f Dockerfile.cpu -t %DOCKER_USERNAME%/%IMAGE_NAME%:cpu .
if %errorlevel% neq 0 exit /b %errorlevel%

echo ✅ Build completed!
echo.
echo Built images:
docker images | findstr %DOCKER_USERNAME%/%IMAGE_NAME%

echo.
echo 🚀 To push to Docker Hub:
echo docker login
echo docker push %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%
echo docker push %DOCKER_USERNAME%/%IMAGE_NAME%:gpu
echo docker push %DOCKER_USERNAME%/%IMAGE_NAME%:cpu

echo.
echo 📋 To run locally:
echo docker run -p 8005:8000 --gpus all %DOCKER_USERNAME%/%IMAGE_NAME%:%VERSION%

pause