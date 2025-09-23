#!/usr/bin/env pwsh

# WhisperX activation script for Windows PowerShell
# This script activates the virtual environment and runs whisperx with the provided arguments
# Supports both audio and video files (video files are converted to audio using ffmpeg)

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

# Get the script directory (where this script is located)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Define the path to the virtual environment python executable
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"

# Check if the virtual environment Python exists
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at: $venvPython"
    Write-Error "Please run setup first: uv pip install --python .\.venv\Scripts\python.exe -e ."
    exit 1
}

# Check if ffmpeg is available
$ffmpegAvailable = $null -ne (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)
if (-not $ffmpegAvailable) {
    Write-Warning "ffmpeg not found in PATH. Video file processing will not be available."
}

# Video file extensions that need audio extraction
$videoExtensions = @('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg')
$audioExtensions = @('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma')

# If no arguments provided, show help
if ($Arguments.Count -eq 0) {
    Write-Host "WhisperX with Video Support" -ForegroundColor Green
    Write-Host "Supports audio files: $($audioExtensions -join ', ')" -ForegroundColor Cyan
    Write-Host "Supports video files: $($videoExtensions -join ', ') (requires ffmpeg)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run_whisperx.ps1 audio.wav --language zh --output_format srt" -ForegroundColor Gray
    Write-Host "  .\run_whisperx.ps1 video.mp4 --language en --output_format vtt" -ForegroundColor Gray
    Write-Host ""
    & $venvPython -m whisperx --help
    exit 0
}

# Parse arguments to find the input file and detect if user specified compute type
$inputFile = $null
$otherArgs = @()
$skipNext = $false
$userSpecifiedComputeType = $false

for ($i = 0; $i -lt $Arguments.Count; $i++) {
    if ($skipNext) {
        $skipNext = $false
        continue
    }
    
    $arg = $Arguments[$i]
    
    # Detect inline form --compute_type=float32
    if ($arg -match '^--compute_type=') {
        $userSpecifiedComputeType = $true
        $otherArgs += $arg
    }
    # Skip known parameters that take values (space separated form)
    elseif ($arg -in @('--model', '--model_dir', '--device', '--device_index', '--batch_size', '--compute_type', 
                   '--output_dir', '-o', '--output_format', '-f', '--language', '--align_model', 
                   '--interpolate_method', '--vad_onset', '--vad_offset', '--chunk_size', '--min_speakers', 
                   '--max_speakers', '--diarize_model', '--temperature', '--best_of', '--beam_size', 
                   '--patience', '--length_penalty', '--suppress_tokens', '--initial_prompt', '--threads', '--hf_token')) {
        $otherArgs += $arg
        if ($i + 1 -lt $Arguments.Count) {
            $otherArgs += $Arguments[$i + 1]
            if ($arg -eq '--compute_type') { $userSpecifiedComputeType = $true }
            $skipNext = $true
        }
    }
    # Skip boolean flags
    elseif ($arg -match '^--') {
        $otherArgs += $arg
    }
    # This should be the input file
    else {
        if ($null -eq $inputFile) {
            $inputFile = $arg
        } else {
            $otherArgs += $arg
        }
    }
}

# Validate input file exists
if ($null -eq $inputFile -or -not (Test-Path $inputFile)) {
    Write-Error "Input file not found or not specified: $inputFile"
    exit 1
}

$inputFile = Resolve-Path $inputFile
$fileExtension = [System.IO.Path]::GetExtension($inputFile).ToLower()
$finalInputFile = $inputFile

# -----------------------------------------------------------------------------
# Automatic Whisper model selection (heuristic) if user did not provide --model
# Can be disabled with --no-auto-model or WHISPERX_DEFAULT_MODEL env var.
# Heuristic (approx memory requirements; assumes int8/float16 where possible):
#   tiny    ~1 GB RAM/VRAM
#   base    ~1.3 GB
#   small   ~2.5 GB
#   medium  ~5.0 GB
#   large-v2/large-v3 ~10-12+ GB (not chosen automatically unless plenty of headroom)
# Selection strategy:
#   1. If WHISPERX_DEFAULT_MODEL env var set -> use it (unless user provided --model)
#   2. If --no-auto-model present -> skip
#   3. Probe hardware (GPU VRAM if CUDA+cuDNN, else system RAM & logical cores)
#   4. Choose largest model that comfortably fits and leaves headroom (factor 1.6x)
#   5. If low resources (<2 GB) -> fall back to base or tiny
# -----------------------------------------------------------------------------

$userSpecifiedModel = $false
$disableAutoModel = $false
$envDefaultModel = $env:WHISPERX_DEFAULT_MODEL

# Detect if user passed --model or --model=<value> or wants to disable auto
for ($k = 0; $k -lt $Arguments.Count; $k++) {
    $a = $Arguments[$k]
    if ($a -eq '--model') { $userSpecifiedModel = $true; break }
    if ($a -like '--model=*') { $userSpecifiedModel = $true; break }
    if ($a -eq '--no-auto-model') { $disableAutoModel = $true }
}

function Select-AutoWhisperModel {
    param(
        [int]$vramMB,
        [int]$systemMemMB,
        [int]$logicalCores,
        [bool]$hasGPU
    )

    # Define candidate models with rough minimum comfortable memory (MB) thresholds
    $models = @(
        @{ name='large-v3'; thresholdMB=18000 },
        @{ name='large-v2'; thresholdMB=16000 },
        @{ name='medium';   thresholdMB=8000 },
        @{ name='small';    thresholdMB=4000 },
        @{ name='base';     thresholdMB=2000 },
        @{ name='tiny';     thresholdMB=1000 }
    )

    if ($hasGPU -and $vramMB -gt 0) {
        foreach ($m in $models) {
            if ($vramMB -ge $m.thresholdMB) { return $m.name }
        }
        # Fallback if very low VRAM
        return 'tiny'
    }
    else {
        # CPU path – scale by system memory and cores
        $effectiveMem = $systemMemMB
        # Light adjustment: if low core count (<4) prefer smaller model to avoid slow runs
        if ($logicalCores -lt 4) { $effectiveMem = [int]($effectiveMem * 0.75) }
        foreach ($m in $models) {
            if ($effectiveMem -ge $m.thresholdMB) { return $m.name }
        }
        return 'tiny'
    }
}

if (-not $userSpecifiedModel -and -not $disableAutoModel) {
    $recommendedModel = $null
    if (-not [string]::IsNullOrWhiteSpace($envDefaultModel)) {
        $recommendedModel = $envDefaultModel
        Write-Host "Using model from WHISPERX_DEFAULT_MODEL='$envDefaultModel'" -ForegroundColor DarkCyan
    }
    else {
        try {
            # Probe hardware via Python for more accurate data (especially GPU VRAM)
            $pyProbe = @'
import json, sys
gpu_vram_mb = 0
cuda = False
try:
    import torch
    if torch.cuda.is_available():
        cuda = True
        try:
            prop = torch.cuda.get_device_properties(0)
            gpu_vram_mb = int(prop.total_memory / (1024*1024))
        except Exception:
            pass
except Exception:
    pass

system_mem_mb = 0
try:
    import psutil
    system_mem_mb = int(psutil.virtual_memory().total / (1024*1024))
except Exception:
    try:
        # Fallback: on Windows use ctypes GlobalMemoryStatusEx, else leave 0
        import ctypes
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memoryLoad", ctypes.c_ulong),
                        ("totalPhys", ctypes.c_ulonglong), ("availPhys", ctypes.c_ulonglong),
                        ("totalPageFile", ctypes.c_ulonglong), ("availPageFile", ctypes.c_ulonglong),
                        ("totalVirtual", ctypes.c_ulonglong), ("availVirtual", ctypes.c_ulonglong),
                        ("availExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUS()
        status.length = ctypes.sizeof(MEMORYSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        system_mem_mb = int(status.totalPhys / (1024*1024))
    except Exception:
        system_mem_mb = 0

try:
    import multiprocessing
    cores = multiprocessing.cpu_count()
except Exception:
    cores = 1

print(json.dumps({
    'cuda': cuda,
    'gpu_vram_mb': gpu_vram_mb,
    'system_mem_mb': system_mem_mb,
    'logical_cores': cores
}))
'@
            $probeJson = & $venvPython -c $pyProbe
            if ($LASTEXITCODE -eq 0 -and $probeJson) {
                $probe = $probeJson | ConvertFrom-Json
                $recommendedModel = Select-AutoWhisperModel -vramMB ([int]$probe.gpu_vram_mb) -systemMemMB ([int]$probe.system_mem_mb) -logicalCores ([int]$probe.logical_cores) -hasGPU ([bool]$probe.cuda)
                Write-Host "Auto-selected Whisper model '$recommendedModel' (GPU=$($probe.cuda) VRAM=$($probe.gpu_vram_mb)MB RAM=$($probe.system_mem_mb)MB Cores=$($probe.logical_cores))." -ForegroundColor DarkCyan
            }
        }
        catch {
            Write-Host "Hardware probe failed; defaulting to 'small' model." -ForegroundColor Yellow
            $recommendedModel = 'small'
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($recommendedModel)) {
        # Insert --model recommendation unless user later overrides (not expected here)
        $otherArgs = @('--model', $recommendedModel) + $otherArgs
    }
}

# Remove the --no-auto-model flag if present so whisperx CLI doesn't choke on it
if ($disableAutoModel) {
    $filtered = @()
    foreach ($a in $otherArgs) { if ($a -ne '--no-auto-model') { $filtered += $a } }
    $otherArgs = $filtered
}

# Auto-detect device and compute type if not specified by user
$userSpecifiedDevice = $false
foreach ($arg in $Arguments) {
    if ($arg -eq '--device' -or $arg -match '^--device=') {
        $userSpecifiedDevice = $true
        break
    }
}

if (-not $userSpecifiedComputeType -or -not $userSpecifiedDevice) {
    try {
        # Check CUDA availability and cuDNN status
        $gpuStatus = & $venvPython -c @"
import torch, sys, json
result = {
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'cudnn_available': hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available() if hasattr(torch.backends, 'cudnn') else False
}
print(json.dumps(result))
"@ 2>$null
        
        $status = $gpuStatus | ConvertFrom-Json
        
        # Check if nvidia-cudnn-cu12 package is actually installed (more reliable than runtime check)
        $cudnnPackageInstalled = & $venvPython -c "
try:
    import nvidia.cudnn
    print('true')
except ImportError:
    print('false')
" 2>$null
        
        # Check if user specified --device cuda or --device=cuda
        $userRequestedCuda = $false
        for ($i = 0; $i -lt $Arguments.Count; $i++) {
            if ($Arguments[$i] -eq '--device' -and ($i + 1) -lt $Arguments.Count -and $Arguments[$i + 1] -eq 'cuda') {
                $userRequestedCuda = $true
                break
            }
            elseif ($Arguments[$i] -match '^--device= *cuda$') {
                $userRequestedCuda = $true
                break
            }
        }
        if ($status.cuda_available -and $status.cuda_device_count -gt 0 -and $cudnnPackageInstalled -eq 'true' -and $userSpecifiedDevice -and $userRequestedCuda) {
            # Only use GPU if user explicitly requested it AND cuDNN package is installed
            Write-Host "CUDA + cuDNN package detected, GPU explicitly requested: using GPU acceleration." -ForegroundColor Green
        }
        elseif ($status.cuda_available -and ($cudnnPackageInstalled -eq 'false' -or -not $userSpecifiedDevice)) {
            # CUDA available but defaulting to CPU for reliability
            if ($cudnnPackageInstalled -eq 'false') {
                Write-Host "CUDA detected but nvidia-cudnn-cu12 package not installed." -ForegroundColor DarkYellow
                Write-Host "Installing cuDNN package for GPU acceleration..." -ForegroundColor Cyan
                
                # Try to auto-install cuDNN
                & uv pip install --python $venvPython nvidia-cudnn-cu12 --quiet
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "cuDNN installed! Use --device cuda to enable GPU acceleration." -ForegroundColor Green
                } else {
                    Write-Host "cuDNN installation failed. Continuing with CPU mode." -ForegroundColor Yellow
                }
            } else {
                Write-Host "CUDA + cuDNN detected but defaulting to CPU mode for reliability." -ForegroundColor DarkYellow
                Write-Host "Use --device cuda to force GPU acceleration." -ForegroundColor Cyan
            }
            Write-Host "Using CPU mode (reliable, works everywhere)." -ForegroundColor Green
            if (-not $userSpecifiedDevice) {
                $otherArgs += '--device'
                $otherArgs += 'cpu'
            }
            if (-not $userSpecifiedComputeType) {
                $otherArgs += '--compute_type'
                $otherArgs += 'float32'
            }
            # Also suggest Silero VAD to avoid pyannote issues
            $vadSpecified = $false
            foreach ($arg in $Arguments) {
                if ($arg -eq '--vad_method' -or $arg -match '^--vad_method=') {
                    $vadSpecified = $true
                    break
                }
            }
            if (-not $vadSpecified) {
                $otherArgs += '--vad_method'
                $otherArgs += 'silero'
                Write-Host "Also switching to Silero VAD to avoid pyannote model compatibility warnings." -ForegroundColor DarkCyan
            }
        }
        else {
            # No CUDA or no GPU devices - pure CPU mode
            if (-not $userSpecifiedComputeType) {
                $otherArgs += '--compute_type'
                $otherArgs += 'float32'
                Write-Host "No CUDA detected: automatically adding --compute_type float32." -ForegroundColor DarkYellow
            }
        }
    }
    catch {
        Write-Warning "Could not auto-detect GPU status; defaulting to CPU-safe settings"
        if (-not $userSpecifiedComputeType) {
            $otherArgs += '--compute_type'
            $otherArgs += 'float32'
        }
        if (-not $userSpecifiedDevice) {
            $otherArgs += '--device'
            $otherArgs += 'cpu'
        }
    }
}

# Determine the effective Whisper model and language for logging
$selectedModel = 'small'
$selectedLanguage = $null

for ($j = 0; $j -lt $otherArgs.Count; $j++) {
    $currentArg = [string]$otherArgs[$j]

    if ($currentArg -eq '--model' -and ($j + 1) -lt $otherArgs.Count) {
        $selectedModel = [string]$otherArgs[$j + 1]
        $j++
        continue
    }

    if ($currentArg -like '--model=*') {
        $selectedModel = $currentArg.Split('=', 2)[1]
        continue
    }

    if ($currentArg -eq '--language' -and ($j + 1) -lt $otherArgs.Count) {
        $selectedLanguage = [string]$otherArgs[$j + 1]
        $j++
        continue
    }

    if ($currentArg -like '--language=*') {
        $selectedLanguage = $currentArg.Split('=', 2)[1]
        continue
    }
}

if ([string]::IsNullOrWhiteSpace($selectedLanguage)) {
    $selectedLanguage = 'auto-detect'
}

Write-Host "Using Whisper model '$selectedModel' for language '$selectedLanguage'." -ForegroundColor Cyan

# Check if input is a video file that needs audio extraction
if ($videoExtensions -contains $fileExtension) {
    if (-not $ffmpegAvailable) {
        Write-Error "Video file detected but ffmpeg is not available. Please install ffmpeg first."
        exit 1
    }
    
    Write-Host "Video file detected: $inputFile" -ForegroundColor Yellow
    Write-Host "Extracting audio using ffmpeg..." -ForegroundColor Cyan
    
    # Create temporary audio file
    $tempAudioFile = [System.IO.Path]::ChangeExtension($inputFile, '.wav')
    $tempAudioFile = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName($inputFile), 
                                               [System.IO.Path]::GetFileNameWithoutExtension($inputFile) + "_temp_audio.wav")
    
    try {
        # Extract audio using ffmpeg
        $ffmpegArgs = @(
            '-i', $inputFile,
            '-vn',                    # No video
            '-acodec', 'pcm_s16le',   # 16-bit PCM
            '-ar', '16000',           # 16kHz sample rate (good for speech recognition)
            '-ac', '1',               # Mono audio
            '-y',                     # Overwrite output file
            $tempAudioFile
        )
        
        Write-Host "Running: ffmpeg $($ffmpegArgs -join ' ')" -ForegroundColor Gray
        & ffmpeg @ffmpegArgs
        
        if ($LASTEXITCODE -ne 0) {
            throw "ffmpeg failed with exit code $LASTEXITCODE"
        }
        
        if (-not (Test-Path $tempAudioFile)) {
            throw "Audio extraction failed - output file not created"
        }
        
        Write-Host "Audio extracted successfully to: $tempAudioFile" -ForegroundColor Green
        $finalInputFile = $tempAudioFile
        
        # Run WhisperX on the extracted audio
        Write-Host "Running WhisperX on extracted audio..." -ForegroundColor Cyan
        & $venvPython -m whisperx $finalInputFile @otherArgs
        
        $whisperExitCode = $LASTEXITCODE
        
        # Clean up temporary audio file
        Write-Host "Cleaning up temporary audio file..." -ForegroundColor Gray
        Remove-Item $tempAudioFile -ErrorAction SilentlyContinue
        
        exit $whisperExitCode
    }
    catch {
        Write-Error "Error processing video file: $($_.Exception.Message)"
        # Clean up temporary audio file on error
        if (Test-Path $tempAudioFile) {
            Remove-Item $tempAudioFile -ErrorAction SilentlyContinue
        }
        exit 1
    }
}
else {
    # Input is already an audio file, process directly
    Write-Host "Audio file detected: $inputFile" -ForegroundColor Green
    & $venvPython -m whisperx $finalInputFile @otherArgs
}
