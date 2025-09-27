# Test GUI functionality by simulating the same process the GUI uses
# This tests the core WhisperX functionality that the GUI calls

param(
    [string]$VideoFile = "D:\whisperX_for_Windows\Hackathon2025.mp4",
    [string]$OutputDir = "D:\whisperX_for_Windows\gui_output",
    [string]$Language = "zh",
    [string]$Format = "srt",
    [string]$Model = "medium"
)

Write-Host "🖥️ Testing GUI Core Functionality" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Simulate GPU detection (same as GUI does)
Write-Host "`n🔍 Probing CUDA/cuDNN support..." -ForegroundColor Cyan
$pythonExe = ".\.venv\Scripts\python.exe"

try {
    $probeScript = @"
import json
result = {}
try:
    import torch
except Exception as exc:
    result['error'] = f'{exc.__class__.__name__}: {exc}'
else:
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        cudnn = getattr(getattr(torch.backends, 'cudnn', None), 'is_available', lambda: False)()
    except Exception as exc:
        result['error'] = f'{exc.__class__.__name__}: {exc}'
    else:
        result['cuda_available'] = bool(cuda_available)
        result['cuda_device_count'] = int(device_count)
        result['cudnn_available'] = bool(cudnn)
print(json.dumps(result))
"@

    $tempScript = [System.IO.Path]::GetTempFileName() + ".py"
    $probeScript | Out-File -FilePath $tempScript -Encoding UTF8
    
    $result = & $pythonExe $tempScript 2>$null | ConvertFrom-Json
    Remove-Item $tempScript -Force
    
    if ($result.error) {
        Write-Host "⚠️ CUDA probe error: $($result.error)" -ForegroundColor Yellow
        $device = "cpu"
        $computeType = "float32"
    }
    elseif ($result.cuda_available -and $result.cudnn_available) {
        Write-Host "✅ CUDA + cuDNN detected: GPU acceleration available" -ForegroundColor Green
        $device = "cuda"
        $computeType = $null
    }
    elseif ($result.cuda_available) {
        Write-Host "⚠️ CUDA detected but cuDNN missing - defaulting to CPU" -ForegroundColor Yellow
        $device = "cpu"
        $computeType = "float32"
    }
    else {
        Write-Host "💻 No CUDA detected - using CPU mode" -ForegroundColor Cyan
        $device = "cpu" 
        $computeType = "float32"
    }
} catch {
    Write-Host "⚠️ GPU probe failed: $($_.Exception.Message)" -ForegroundColor Yellow
    $device = "cpu"
    $computeType = "float32"
}

# Build arguments (same as GUI does)
Write-Host "`n📋 Building WhisperX arguments..." -ForegroundColor Cyan
$args = @(
    "-m", "whisperx",
    $VideoFile,
    "--model", $Model,
    "-o", $OutputDir,
    "-f", $Format,
    "--language", $Language,
    "--print_progress", "True"
)

if ($device -eq "cpu") {
    $args += "--device", "cpu"
    $args += "--compute_type", $computeType
    Write-Host "🖥️ CPU device selected: forcing --compute_type $computeType" -ForegroundColor Yellow
}

Write-Host "`n🎬 Processing video file..." -ForegroundColor Cyan
Write-Host "Input: $VideoFile" -ForegroundColor Gray
Write-Host "Output: $OutputDir" -ForegroundColor Gray  
Write-Host "Language: $Language" -ForegroundColor Gray
Write-Host "Format: $Format" -ForegroundColor Gray
Write-Host "Model: $Model" -ForegroundColor Gray
Write-Host "Device: $device" -ForegroundColor Gray

# Run WhisperX (same as GUI does)
Write-Host "`n▶️ Running WhisperX..." -ForegroundColor Green
Write-Host "> Command: $pythonExe $($args -join ' ')" -ForegroundColor DarkGray

try {
    $startTime = Get-Date
    & $pythonExe @args
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Processing completed successfully!" -ForegroundColor Green
        Write-Host "⏱️ Processing time: $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
        
        # Check output files
        $outputFiles = Get-ChildItem -Path $OutputDir -Filter "*.srt" -ErrorAction SilentlyContinue
        if ($outputFiles) {
            Write-Host "`n📄 Generated files:" -ForegroundColor Cyan
            foreach ($file in $outputFiles) {
                Write-Host "  • $($file.Name) ($($file.Length) bytes)" -ForegroundColor White
            }
            
            # Show first few lines of the subtitle file
            $firstFile = $outputFiles[0]
            Write-Host "`n📝 Subtitle content preview:" -ForegroundColor Cyan
            Get-Content $firstFile.FullName -Head 10 | ForEach-Object {
                Write-Host "  $_" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "`n❌ Processing failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n❌ Processing failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n🏁 GUI functionality test completed." -ForegroundColor Green