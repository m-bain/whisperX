# Windows Setup & Troubleshooting Guide

## 🚀 One-Command Setup Script

Create this PowerShell script to automate the entire setup process:

```powershell
# setup_whisperx.ps1 - Run as Administrator
Write-Host "🚀 WhisperX for Windows - Automated Setup" -ForegroundColor Green

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "Please run this script as Administrator!"
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Install prerequisites via winget
Write-Host "📦 Installing prerequisites..." -ForegroundColor Cyan
$packages = @(
    "Python.Python.3.11",
    "astral-sh.uv", 
    "Gyan.FFmpeg",
    "Microsoft.DotNet.SDK.8"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Yellow
    winget install -e --id $package --scope machine --silent
}

# Refresh PATH environment variable
Write-Host "🔄 Refreshing environment variables..." -ForegroundColor Cyan
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Setup Python environment
Write-Host "🐍 Setting up Python environment..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
}
uv venv
uv pip install -e .

# Test installation
Write-Host "✅ Testing installation..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m whisperx --version
ffmpeg -version | Select-String "ffmpeg version"
dotnet --version

Write-Host "🎉 Setup complete! Ready to use WhisperX." -ForegroundColor Green
Write-Host "Try: .\run_whisperx.ps1 your_video.mp4 --language en --output_format srt" -ForegroundColor Yellow
```

## 🛠 Step-by-Step Manual Setup

### 1. Prerequisites Installation

**Method 1: Using winget (Recommended)**
```powershell
# Open PowerShell as Administrator
winget install -e --id Python.Python.3.11 --scope machine
winget install -e --id astral-sh.uv --scope machine  
winget install -e --id Gyan.FFmpeg --scope machine
winget install -e --id Microsoft.DotNet.SDK.8 --scope machine
```

**Method 2: Direct Downloads**
- [Python 3.11+](https://www.python.org/downloads/windows/) - Add to PATH during installation
- [UV Package Manager](https://docs.astral.sh/uv/getting-started/installation/) - Modern Python package manager
- [FFmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) - For video processing
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) - For GUI application

### 2. Environment Setup

```powershell
# Navigate to project directory
cd D:\whisperX_for_Windows

# Create and setup Python environment
uv venv
uv pip install -e .

# Verify installation
.\.venv\Scripts\python.exe -m whisperx --help
```

### 3. First Run Test

```powershell
# Test with sample (creates output directory automatically)
.\run_whisperx.ps1 sample_video.mp4 --language en --output_format srt

# Or start GUI
dotnet run --project WhisperXGUI/WhisperXGUI.csproj
```

## 🩺 Common Issues & Solutions

### Issue: "uv: command not found"
**Solution:**
```powershell
# Refresh PATH or restart PowerShell
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Alternative: Use pip
pip install uv
```

### Issue: "ffmpeg not found in PATH"
**Symptoms:** `Video file detected but ffmpeg is not available`

**Solution:**
```powershell
# Option 1: Reinstall via winget
winget uninstall Gyan.FFmpeg
winget install -e --id Gyan.FFmpeg --scope machine

# Option 2: Manual PATH addition
$ffmpegPath = "C:\ffmpeg\bin"  # Adjust path as needed
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";" + $ffmpegPath, "Machine")

# Option 3: Verify installation location
where.exe ffmpeg
```

### Issue: Virtual Environment Activation
**Symptoms:** `ModuleNotFoundError: No module named 'torch'`

**Solutions:**
```powershell
# Always use full path to venv python
.\.venv\Scripts\python.exe -m whisperx --help

# Or activate environment first
.\.venv\Scripts\Activate.ps1
python -m whisperx --help

# Set environment variable for consistency
$env:WHISPERX_PYTHON = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

### Issue: Long Download Times (First Run)
**Expected:** Large model downloads (1-5 GB) on first use

**Solutions:**
```powershell
# Pre-download models (optional)
.\.venv\Scripts\python.exe -c "import whisperx; whisperx.load_model('medium')"

# Use smaller model for testing
.\run_whisperx.ps1 video.mp4 --model tiny --language en --output_format srt

# Show progress during processing
.\run_whisperx.ps1 video.mp4 --language en --output_format srt --verbose True
```

### Issue: GPU/CUDA Errors
**Symptoms:** CUDA-related errors or poor performance

**Solutions:**
```powershell
# Force CPU mode (most reliable)
.\run_whisperx.ps1 video.mp4 --device cpu --compute_type float32

# Check GPU status
nvidia-smi

# Install CUDA support (optional)
winget install -e --id Nvidia.CUDA
uv pip install --python .\.venv\Scripts\python.exe nvidia-cudnn-cu12
```

### Issue: Permission Errors
**Symptoms:** Access denied or file locked errors

**Solutions:**
```powershell
# Run PowerShell as Administrator for initial setup
# Ensure antivirus isn't blocking model downloads
# Check disk space (models require several GB)

# Alternative output location
.\run_whisperx.ps1 video.mp4 --output_dir C:\Temp\whisperx_output
```

## 📁 Directory Structure After Setup

```
whisperX_for_Windows/
├── .venv/                    # Python virtual environment
│   └── Scripts/
│       └── python.exe       # Use this for consistent execution
├── output/                   # Default output directory (auto-created)
├── whisperx/                 # Core Python package
├── WhisperXGUI/             # .NET WPF application
│   └── bin/Release/         # Built GUI executable
├── run_whisperx.ps1         # Main PowerShell script
└── setup_whisperx.ps1       # Setup automation script (create this)
```

## 🎯 Performance Optimization

### For Better Performance:
```powershell
# Use appropriate model for your hardware
.\run_whisperx.ps1 video.mp4 --model small    # 4GB+ RAM
.\run_whisperx.ps1 video.mp4 --model medium   # 8GB+ RAM  
.\run_whisperx.ps1 video.mp4 --model large-v2 # 16GB+ RAM

# Optimize for CPU
.\run_whisperx.ps1 video.mp4 --compute_type int8 --device cpu

# Skip alignment for faster processing (less precise timestamps)
.\run_whisperx.ps1 video.mp4 --no_align
```

### For Better Accuracy:
```powershell
# Enable speaker diarization
.\run_whisperx.ps1 video.mp4 --diarize --max_speakers 3

# Use Silero VAD for better silence detection  
.\run_whisperx.ps1 video.mp4 --vad_method silero
```

## 🔧 Environment Variables

Set these for consistent behavior:

```powershell
# Set default model
$env:WHISPERX_DEFAULT_MODEL = 'medium'

# Set Python path (avoid activation issues)
$env:WHISPERX_PYTHON = 'D:\whisperX_for_Windows\.venv\Scripts\python.exe'

# Make permanent
[Environment]::SetEnvironmentVariable("WHISPERX_DEFAULT_MODEL", "medium", "User")
[Environment]::SetEnvironmentVariable("WHISPERX_PYTHON", "D:\whisperX_for_Windows\.venv\Scripts\python.exe", "User")
```

## 📝 Usage Examples

### Basic Transcription
```powershell
# English video to SRT
.\run_whisperx.ps1 meeting.mp4 --language en --output_format srt

# Chinese video to multiple formats
.\run_whisperx.ps1 lecture.mp4 --language zh --output_format all

# Auto-detect language
.\run_whisperx.ps1 video.mp4 --output_format srt
```

### Advanced Features
```powershell
# Multi-speaker video with diarization
.\run_whisperx.ps1 panel.mp4 --diarize --max_speakers 4 --language en

# High-accuracy with specific model
.\run_whisperx.ps1 interview.wav --model large-v2 --language en --output_format vtt

# Batch processing multiple files
Get-ChildItem "*.mp4" | ForEach-Object {
    .\run_whisperx.ps1 $_.Name --language en --output_format srt --output_dir subtitles
}
```

## 🚨 Firewall & Antivirus Notes

- **Model Downloads**: WhisperX downloads large AI models from Hugging Face (2-5GB)
- **Allow Network Access**: First run requires internet for model downloads
- **Folder Exclusions**: Add project folder to antivirus exclusions if downloads fail
- **Temporary Files**: Models cache in `~/.cache/huggingface/` and `~/.cache/whisper/`

## 📞 Getting Help

If issues persist:

1. **Check Prerequisites**: Ensure all dependencies installed correctly
2. **Test Components**: Verify Python, FFmpeg, and .NET individually  
3. **Clean Reinstall**: Delete `.venv` folder and run setup again
4. **Check Logs**: Review error messages for specific guidance
5. **Use CPU Mode**: Add `--device cpu --compute_type float32` for stability

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.11+ installed and accessible
- [ ] UV package manager working: `uv --version`
- [ ] FFmpeg available: `ffmpeg -version`
- [ ] .NET 8 SDK installed: `dotnet --version`
- [ ] Virtual environment created: `.venv` folder exists
- [ ] WhisperX CLI working: `.\.venv\Scripts\python.exe -m whisperx --help`
- [ ] PowerShell script executable: `.\run_whisperx.ps1` (shows help)
- [ ] Sufficient disk space (5-10GB for models and processing)

This setup process ensures a smooth Windows experience with WhisperX!