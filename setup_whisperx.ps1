# WhisperX for Windows - Automated Setup Script
# Run as Administrator: Right-click PowerShell -> "Run as Administrator"

param(
    [switch]$SkipGPU,
    [switch]$Force
)

Write-Host @"
🚀 WhisperX for Windows - Automated Setup
==========================================
This script will install all prerequisites and set up WhisperX.
Run this script as Administrator for best results.

"@ -ForegroundColor Green

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Warning @"
⚠️  Not running as Administrator!
   For automatic installation, please:
   1. Right-click PowerShell
   2. Select 'Run as Administrator'
   3. Run this script again
   
   Continuing with limited functionality...
"@
    Start-Sleep -Seconds 3
}

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Check winget availability
$hasWinget = Test-CommandExists "winget"
if (-not $hasWinget) {
    Write-Warning "winget not available. Please install manually from Microsoft Store or GitHub."
}

Write-Host "📦 Installing Prerequisites..." -ForegroundColor Cyan

# Define packages to install
$packages = @(
    @{ id = "Python.Python.3.11"; name = "Python 3.11"; command = "python" },
    @{ id = "astral-sh.uv"; name = "UV Package Manager"; command = "uv" },
    @{ id = "Gyan.FFmpeg"; name = "FFmpeg"; command = "ffmpeg" },
    @{ id = "Microsoft.DotNet.SDK.8"; name = ".NET 8 SDK"; command = "dotnet" }
)

# Optionally install GPU support
if (-not $SkipGPU) {
    $packages += @{ id = "Nvidia.CUDA"; name = "NVIDIA CUDA Toolkit"; command = "nvcc" }
}

foreach ($package in $packages) {
    Write-Host "Checking $($package.name)..." -ForegroundColor Yellow
    
    if (Test-CommandExists $package.command) {
        Write-Host "✅ $($package.name) already installed" -ForegroundColor Green
        continue
    }
    
    if ($hasWinget -and $isAdmin) {
        Write-Host "Installing $($package.name)..." -ForegroundColor Yellow
        try {
            winget install -e --id $package.id --scope machine --silent --accept-package-agreements --accept-source-agreements
            Write-Host "✅ $($package.name) installed successfully" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to install $($package.name): $($_.Exception.Message)"
        }
    } else {
        Write-Warning "❌ Please install $($package.name) manually"
        switch ($package.id) {
            "Python.Python.3.11" { Write-Host "   Download from: https://www.python.org/downloads/windows/" -ForegroundColor Gray }
            "astral-sh.uv" { Write-Host "   Install with: pip install uv" -ForegroundColor Gray }
            "Gyan.FFmpeg" { Write-Host "   Download from: https://github.com/BtbN/FFmpeg-Builds/releases" -ForegroundColor Gray }
            "Microsoft.DotNet.SDK.8" { Write-Host "   Download from: https://dotnet.microsoft.com/download/dotnet/8.0" -ForegroundColor Gray }
            "Nvidia.CUDA" { Write-Host "   Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Gray }
        }
    }
}

Write-Host "`n🔄 Refreshing Environment Variables..." -ForegroundColor Cyan
# Refresh PATH environment variable
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# Wait a moment for environment to refresh
Start-Sleep -Seconds 2

Write-Host "`n🐍 Setting up Python Environment..." -ForegroundColor Cyan

# Remove existing venv if Force is specified
if ($Force -and (Test-Path ".venv")) {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

# Check if UV is available
if (Test-CommandExists "uv") {
    Write-Host "Creating virtual environment with UV..." -ForegroundColor Yellow
    try {
        uv venv
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        uv pip install -e .
        Write-Host "✅ Python environment setup complete" -ForegroundColor Green
    } catch {
        Write-Error "Failed to setup Python environment with UV: $($_.Exception.Message)"
        exit 1
    }
} else {
    Write-Warning "UV not available. Please install manually with: pip install uv"
    exit 1
}

Write-Host "`n✅ Testing Installation..." -ForegroundColor Cyan

# Test components
$tests = @{
    "Python Environment" = { .\.venv\Scripts\python.exe --version }
    "WhisperX CLI" = { .\.venv\Scripts\python.exe -m whisperx --version }
    "FFmpeg" = { ffmpeg -version 2>$null | Select-String "ffmpeg version" | Select-Object -First 1 }
    ".NET SDK" = { dotnet --version }
    "PowerShell Script" = { .\run_whisperx.ps1 2>&1 | Select-String "WhisperX with Video Support" }
}

$allPassed = $true

foreach ($test in $tests.GetEnumerator()) {
    try {
        $result = & $test.Value 2>$null
        if ($result) {
            Write-Host "✅ $($test.Key): Working" -ForegroundColor Green
        } else {
            Write-Host "❌ $($test.Key): Failed" -ForegroundColor Red
            $allPassed = $false
        }
    } catch {
        Write-Host "❌ $($test.Key): Error - $($_.Exception.Message)" -ForegroundColor Red
        $allPassed = $false
    }
}

# Create output directory
if (-not (Test-Path "output")) {
    New-Item -ItemType Directory -Path "output" | Out-Null
    Write-Host "📁 Created output directory" -ForegroundColor Cyan
}

# Set helpful environment variables
Write-Host "`n🔧 Setting Environment Variables..." -ForegroundColor Cyan
$pythonPath = (Resolve-Path ".\.venv\Scripts\python.exe").Path
[Environment]::SetEnvironmentVariable("WHISPERX_PYTHON", $pythonPath, "User")
[Environment]::SetEnvironmentVariable("WHISPERX_DEFAULT_MODEL", "small", "User")
Write-Host "Set WHISPERX_PYTHON = $pythonPath" -ForegroundColor Gray
Write-Host "Set WHISPERX_DEFAULT_MODEL = small" -ForegroundColor Gray

Write-Host "`n" -NoNewline
if ($allPassed) {
    Write-Host @"
🎉 Setup Complete! WhisperX is ready to use.

📝 Quick Start Examples:
   # Generate English subtitles
   .\run_whisperx.ps1 your_video.mp4 --language en --output_format srt

   # Generate Chinese subtitles  
   .\run_whisperx.ps1 your_video.mp4 --language zh --output_format srt

   # Launch GUI application
   dotnet run --project WhisperXGUI/WhisperXGUI.csproj

💡 Tips:
   - First run downloads models (2-5GB) - be patient!
   - Models are cached for faster subsequent runs
   - Use --device cpu --compute_type float32 for stability
   - See WINDOWS_SETUP_GUIDE.md for detailed help

"@ -ForegroundColor Green
} else {
    Write-Host @"
⚠️  Setup completed with some issues.
    Please review the failed tests above and:
    1. Install missing components manually
    2. Check WINDOWS_SETUP_GUIDE.md for troubleshooting
    3. Re-run this script with -Force to retry

"@ -ForegroundColor Yellow
}

Write-Host "Press any key to continue..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")