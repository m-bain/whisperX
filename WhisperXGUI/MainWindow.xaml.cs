using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Media;
using System.Timers;
using WinForms = System.Windows.Forms;

namespace WhisperXGUI;

public enum LogLevel
{
    Debug,
    Info,
    Success,
    Warning,
    Error,
    Progress
}

public partial class MainWindow : Window, INotifyPropertyChanged
{
    private string _inputFile = string.Empty;
    private string _outputFolder = string.Empty;
    private string _logText = string.Empty;
    private string _status = "Idle";
    private bool _isRunning;
    private string _selectedModel = "small";
    private string _selectedFormat = "srt";
    private string _selectedDevice = "auto";
    private bool _diarize;
    private string _languageOverride = string.Empty;
    
    // Enhanced properties for better UX
    private string _currentStep = "Ready";
    private string _progressText = "";
    private double _progressValue = 0;
    private bool _isIndeterminate = false;
    private DateTime _startTime;
    private System.Timers.Timer? _elapsedTimer;
    private string _fileInfoText = "";
    private string _processingEstimate = "";
    private string _modelInfo = "";
    private string _hardwareInfo = "";
    private string _systemInfo = "";

    private CancellationTokenSource? _cts;
    private TorchGpuStatus? _cachedGpuStatus;
    private const string TorchProbeScript = """
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
""";

    public ObservableCollection<string> Models { get; } = new(["tiny","base","small","medium","large-v2"]);
    public ObservableCollection<string> Formats { get; } = new(["srt","vtt","txt","tsv","json","aud"]);
    public ObservableCollection<string> Devices { get; } = new(["auto","cpu"]);

    // Basic properties
    public string InputFile { get => _inputFile; set { _inputFile = value; OnPropertyChanged(); UpdateCanRun(); UpdateFileInfo(); } }
    public string OutputFolder { get => _outputFolder; set { _outputFolder = value; OnPropertyChanged(); UpdateCanRun(); } }
    public string LogText { get => _logText; set { _logText = value; OnPropertyChanged(); } }
    public string Status { get => _status; set { _status = value; OnPropertyChanged(); } }
    public bool IsRunning { get => _isRunning; set { _isRunning = value; OnPropertyChanged(); OnPropertyChanged(nameof(CanRun)); } }
    public string SelectedModel { get => _selectedModel; set { _selectedModel = value; OnPropertyChanged(); UpdateModelInfo(); } }
    public string SelectedFormat { get => _selectedFormat; set { _selectedFormat = value; OnPropertyChanged(); } }
    public string SelectedDevice { get => _selectedDevice; set { _selectedDevice = value; OnPropertyChanged(); } }
    public bool Diarize { get => _diarize; set { _diarize = value; OnPropertyChanged(); } }
    public string LanguageOverride { get => _languageOverride; set { _languageOverride = value; OnPropertyChanged(); } }

    // Enhanced properties
    public string CurrentStep { get => _currentStep; set { _currentStep = value; OnPropertyChanged(); } }
    public string ProgressText { get => _progressText; set { _progressText = value; OnPropertyChanged(); } }
    public double ProgressValue { get => _progressValue; set { _progressValue = value; OnPropertyChanged(); OnPropertyChanged(nameof(ProgressPercent)); } }
    public bool IsIndeterminate { get => _isIndeterminate; set { _isIndeterminate = value; OnPropertyChanged(); } }
    public string ProgressPercent => $"{ProgressValue:F1}%";
    public string FileInfoText { get => _fileInfoText; set { _fileInfoText = value; OnPropertyChanged(); OnPropertyChanged(nameof(HasFileInfo)); } }
    public string ProcessingEstimate { get => _processingEstimate; set { _processingEstimate = value; OnPropertyChanged(); } }
    public string ModelInfo { get => _modelInfo; set { _modelInfo = value; OnPropertyChanged(); } }
    public string HardwareInfo { get => _hardwareInfo; set { _hardwareInfo = value; OnPropertyChanged(); } }
    public string SystemInfo { get => _systemInfo; set { _systemInfo = value; OnPropertyChanged(); } }
    public bool HasFileInfo => !string.IsNullOrEmpty(_fileInfoText);
    
    public string ElapsedTime
    {
        get
        {
            if (_startTime == default) return "00:00:00";
            var elapsed = DateTime.Now - _startTime;
            return elapsed.ToString(@"hh\:mm\:ss");
        }
    }
    
    public string EstimatedTimeRemaining
    {
        get
        {
            if (_startTime == default || ProgressValue <= 0) return "Calculating...";
            var elapsed = DateTime.Now - _startTime;
            var estimated = TimeSpan.FromMilliseconds(elapsed.TotalMilliseconds * (100.0 / ProgressValue - 1.0));
            return estimated.TotalHours >= 1 
                ? estimated.ToString(@"hh\:mm\:ss")
                : estimated.ToString(@"mm\:ss");
        }
    }

    public bool CanRun => !IsRunning && File.Exists(InputFile) && Directory.Exists(OutputFolder);

    public event PropertyChangedEventHandler? PropertyChanged;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        OutputFolder = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        
        // Initialize enhanced features
        InitializeSystemInfo();
        UpdateModelInfo();
        
        Loaded += MainWindow_Loaded;
    }

    private void InitializeSystemInfo()
    {
        var totalRam = GetSystemMemoryGB();
        var cores = Environment.ProcessorCount;
        SystemInfo = $"System: {totalRam:F1}GB RAM, {cores} cores";
        
        // Initialize timer for elapsed time updates
        _elapsedTimer = new System.Timers.Timer(1000);
        _elapsedTimer.Elapsed += (s, e) => 
        {
            Dispatcher.Invoke(() => 
            {
                OnPropertyChanged(nameof(ElapsedTime));
                OnPropertyChanged(nameof(EstimatedTimeRemaining));
            });
        };
    }

    private double GetSystemMemoryGB()
    {
        try
        {
            var gc = GC.GetTotalMemory(false);
            var workingSet = Environment.WorkingSet;
            // This is an approximation - in a real app you'd use WMI or similar
            return Math.Max(8.0, workingSet / (1024.0 * 1024.0 * 1024.0) * 4); // Rough estimate
        }
        catch
        {
            return 16.0; // Default assumption
        }
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        Loaded -= MainWindow_Loaded;
        AppendLog("🚀 WhisperX Enhanced GUI initialized", LogLevel.Success);
        AppendLog("🔍 Detecting hardware capabilities...", LogLevel.Info);
        await DetectCudaSupportAsync();
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private void UpdateCanRun() => OnPropertyChanged(nameof(CanRun));

    private void Model_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        UpdateModelInfo();
        UpdateFileInfo(); // Re-calculate processing estimate
    }

    private void BrowseFile_Click(object sender, RoutedEventArgs e)
    {
        var ofd = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Media Files|*.mp4;*.mkv;*.wav;*.mp3;*.flac;*.ogg;*.mov;*.avi;*.wmv;*.flv;*.webm;*.m4v;*.3gp;*.mpg;*.mpeg;*.aac;*.m4a;*.wma|All Files|*.*",
            CheckFileExists = true,
            Title = "Select Media File for Transcription"
        };

        if (ofd.ShowDialog() == true)
        {
            InputFile = ofd.FileName;

            var parent = Path.GetDirectoryName(InputFile);
            if ((string.IsNullOrWhiteSpace(OutputFolder) || !Directory.Exists(OutputFolder)) && !string.IsNullOrEmpty(parent))
            {
                OutputFolder = parent;
            }
            
            AppendLog($"📁 Selected file: {Path.GetFileName(InputFile)}", LogLevel.Info);
        }
    }

    private void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        using var fbd = new WinForms.FolderBrowserDialog();
        fbd.Description = "Select Output Folder for Generated Subtitles";
        fbd.UseDescriptionForTitle = true;
        
        if (fbd.ShowDialog() == WinForms.DialogResult.OK && !string.IsNullOrEmpty(fbd.SelectedPath))
        {
            OutputFolder = fbd.SelectedPath;
            AppendLog($"📂 Output folder: {OutputFolder}", LogLevel.Info);
        }
    }

    private async void Run_Click(object sender, RoutedEventArgs e)
    {
        if (!CanRun) return;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        Status = "Processing...";
        CurrentStep = "Initializing";
        ProgressValue = 0;
        IsIndeterminate = true;
        
        _startTime = DateTime.Now;
        _elapsedTimer?.Start();

        AppendLog("🚀 Starting WhisperX processing...", LogLevel.Info);
        
        try
        {
            await RunWhisperXAsync(_cts.Token);
            Status = "Completed Successfully";
            AppendLog("🎉 Processing completed successfully!", LogLevel.Success);
            ProgressValue = 100;
            IsIndeterminate = false;
        }
        catch (OperationCanceledException)
        {
            Status = "Canceled";
            AppendLog("🛑 Operation canceled by user", LogLevel.Warning);
            CurrentStep = "Canceled";
        }
        catch (Exception ex)
        {
            Status = "Error";
            AppendLog($"❌ Processing failed: {ex.Message}", LogLevel.Error);
            CurrentStep = "Failed";
        }
        finally
        {
            IsRunning = false;
            _elapsedTimer?.Stop();
            
            var elapsed = DateTime.Now - _startTime;
            AppendLog($"⏱️ Total processing time: {elapsed:mm\\:ss}", LogLevel.Info);
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        AppendLog("🛑 Cancellation requested...", LogLevel.Warning);
        _cts?.Cancel();
    }
    
    // Enhanced log management methods
    private void CopyLog_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var logContent = new TextRange(LogDisplay.Document.ContentStart, LogDisplay.Document.ContentEnd).Text;
            System.Windows.Clipboard.SetText(logContent);
            AppendLog("📋 Log copied to clipboard", LogLevel.Info);
        }
        catch (Exception ex)
        {
            AppendLog($"❌ Failed to copy log: {ex.Message}", LogLevel.Error);
        }
    }
    
    private void SaveLog_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var sfd = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "Text Files|*.txt|Log Files|*.log|All Files|*.*",
                DefaultExt = "txt",
                FileName = $"WhisperX_Log_{DateTime.Now:yyyyMMdd_HHmmss}.txt"
            };
            
            if (sfd.ShowDialog() == true)
            {
                var logContent = new TextRange(LogDisplay.Document.ContentStart, LogDisplay.Document.ContentEnd).Text;
                File.WriteAllText(sfd.FileName, logContent);
                AppendLog($"💾 Log saved to: {Path.GetFileName(sfd.FileName)}", LogLevel.Success);
            }
        }
        catch (Exception ex)
        {
            AppendLog($"❌ Failed to save log: {ex.Message}", LogLevel.Error);
        }
    }
    
    private void ClearLog_Click(object sender, RoutedEventArgs e)
    {
        LogDisplay.Document.Blocks.Clear();
        AppendLog("🗑️ Log cleared", LogLevel.Info);
    }

    private async Task RunWhisperXAsync(CancellationToken token)
    {
        var pythonExe = ResolvePython();
        if (string.IsNullOrEmpty(pythonExe))
            throw new InvalidOperationException("Python executable not found or not usable. Install Python 3.10+ or set WHISPERX_PYTHON.");

        AppendLog($"🐍 Using Python: {pythonExe}", LogLevel.Debug);
        
        // Log file information
        if (File.Exists(InputFile))
        {
            var fileInfo = new FileInfo(InputFile);
            var extension = fileInfo.Extension.ToLower();
            var isVideo = new[] { ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".3gp", ".mpg", ".mpeg" }
                .Contains(extension);
                
            AppendLog($"📁 Processing: {fileInfo.Name} ({FormatFileSize(fileInfo.Length)})", LogLevel.Info);
            if (isVideo)
            {
                AppendLog("🎬 Video file detected - will extract audio using FFmpeg", LogLevel.Info);
            }
        }

        var argumentList = await BuildWhisperArgumentListAsync(pythonExe, token).ConfigureAwait(false);

        AppendLog($"🔧 Command: {pythonExe} {string.Join(' ', argumentList.Select(RenderArgumentForLog))}", LogLevel.Debug);

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = GetWhisperXWorkingDirectory()
        };

        foreach (var argument in argumentList)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        var tcs = new TaskCompletionSource<int>();

        // Enhanced output handling
        process.OutputDataReceived += (_, e) => 
        { 
            if (e.Data != null) 
                HandleProcessOutput(e.Data); 
        };
        process.ErrorDataReceived += (_, e) => 
        { 
            if (e.Data != null) 
                HandleProcessOutput(e.Data); 
        };
        process.Exited += (_, _) => tcs.TrySetResult(process.ExitCode);

        if (!process.Start())
            throw new InvalidOperationException("Failed to start whisperx process");

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using (token.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                {
                    AppendLog("🛑 Terminating process...", LogLevel.Warning);
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // ignored
            }
        }))
        {
            int exitCode = await tcs.Task.ConfigureAwait(false);
            if (exitCode != 0)
                throw new InvalidOperationException($"whisperx exited with code {exitCode}");
        }

        CurrentStep = "Completed";
        AppendLog("✅ WhisperX processing completed successfully", LogLevel.Success);
    }
    
    private void HandleProcessOutput(string output)
    {
        if (string.IsNullOrWhiteSpace(output)) return;

        // Handle special WhisperX output patterns with enhanced parsing
        if (output.Contains("model.safetensors:") && output.Contains("%"))
        {
            // Model download progress - already handled in EnhanceMessage
            AppendLog(output, LogLevel.Progress);
            return;
        }
        
        if (output.Contains("Lightning automatically upgraded"))
        {
            AppendLog("⚡ Model compatibility: Upgrading checkpoint format", LogLevel.Info);
            return;
        }
        
        if (output.Contains("Model was trained with") && output.Contains("yours is"))
        {
            AppendLog("⚠️ Model version compatibility warning (non-critical)", LogLevel.Warning);
            return;
        }
        
        if (output.Contains("UserWarning") && (output.Contains("deprecated") || output.Contains("will be removed")))
        {
            // Suppress verbose deprecation warnings
            return;
        }
        
        // Handle transcript output with enhanced formatting
        if (output.Contains("Transcript:"))
        {
            AppendLog(output, LogLevel.Success);
            return;
        }
        
        // Handle progress indicators
        if (output.Contains("Progress: 100.00%"))
        {
            ProgressValue = 100;
            IsIndeterminate = false;
            AppendLog("📊 Step completed (100%)", LogLevel.Success);
            return;
        }
        
        // Default handling with smart level detection
        AppendLog(output);
    }

    private async Task DetectCudaSupportAsync()
    {
        var pythonExe = ResolvePython();
        if (string.IsNullOrEmpty(pythonExe))
        {
            AppendLog("⚠️ Python not found - cannot detect GPU capabilities", LogLevel.Warning);
            HardwareInfo = "⚠️ Python not detected - using CPU mode";
            return;
        }

        AppendLog($"🔍 Probing hardware via {Path.GetFileName(pythonExe)}", LogLevel.Info);
        
        try
        {
            var status = await ProbeTorchGpuStatusAsync(pythonExe, CancellationToken.None);
            _cachedGpuStatus = status;

            if (status.HasError)
            {
                AppendLog("⚠️ GPU detection failed - defaulting to CPU mode", LogLevel.Warning);
                if (!string.IsNullOrWhiteSpace(status.ErrorMessage))
                {
                    AppendLog($"🔍 Details: {status.ErrorMessage}", LogLevel.Debug);
                }
                HardwareInfo = "🖥️ CPU mode (GPU detection failed)";
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                return;
            }

            if (!string.IsNullOrWhiteSpace(status.InfoMessage))
            {
                AppendLog(status.InfoMessage, LogLevel.Info);
            }

            if (status.HasCuda && status.HasCudnn)
            {
                if (!Devices.Contains("cuda"))
                {
                    Devices.Add("cuda");
                }
                AppendLog("🎮 GPU acceleration available (CUDA + cuDNN)", LogLevel.Success);
                AppendLog($"🔧 GPU devices: {status.DeviceCount}", LogLevel.Debug);
                HardwareInfo = $"🎮 GPU ready ({status.DeviceCount} device{(status.DeviceCount > 1 ? "s" : "")})";
            }
            else if (status.CudaAvailable == true && !status.HasCudnn)
            {
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                AppendLog("🎮 CUDA detected but cuDNN missing", LogLevel.Warning);
                AppendLog("💡 Install cuDNN for GPU acceleration: uv pip install nvidia-cudnn-cu12", LogLevel.Info);
                HardwareInfo = "⚠️ CUDA available (cuDNN needed for GPU)";
            }
            else
            {
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                AppendLog("🖥️ No GPU acceleration - using CPU mode", LogLevel.Info);
                AppendLog("💡 For faster processing, consider a CUDA-compatible GPU", LogLevel.Debug);
                HardwareInfo = "🖥️ CPU mode (no CUDA GPU detected)";
            }
            
            // Log system information
            var systemRam = GetSystemMemoryGB();
            AppendLog($"💾 System: {systemRam:F1}GB RAM, {Environment.ProcessorCount} CPU cores", LogLevel.Debug);
            
        }
        catch (Exception ex)
        {
            AppendLog($"❌ Hardware detection failed: {ex.Message}", LogLevel.Error);
            HardwareInfo = "❌ Hardware detection failed";
        }
    }

    private async Task<List<string>> BuildWhisperArgumentListAsync(string pythonExe, CancellationToken token)
    {
        var args = new List<string>
        {
            "-m",
            "whisperx",
            InputFile
        };

        args.Add("--model");
        args.Add(SelectedModel);
        args.Add("-o");
        args.Add(OutputFolder);
        args.Add("-f");
        args.Add(SelectedFormat);

        if (Diarize)
        {
            args.Add("--diarize");
        }

        if (!string.IsNullOrWhiteSpace(LanguageOverride))
        {
            args.Add("--language");
            args.Add(LanguageOverride);
        }

        args.Add("--print_progress");
        args.Add("True");

        var languageForLog = string.IsNullOrWhiteSpace(LanguageOverride)
            ? "auto-detect"
            : LanguageOverride;

        if (string.Equals(SelectedDevice, "cpu", StringComparison.OrdinalIgnoreCase))
        {
            args.Add("--device");
            args.Add("cpu");
            args.Add("--compute_type");
            args.Add("float32");
            AppendLog("CPU device selected: forcing --compute_type float32.");
        }
        else if (string.Equals(SelectedDevice, "cuda", StringComparison.OrdinalIgnoreCase))
        {
            args.Add("--device");
            args.Add("cuda");
        }
        else
        {
            var status = await ProbeTorchGpuStatusAsync(pythonExe, token).ConfigureAwait(false);
            _cachedGpuStatus = status;

            var decision = BuildAutoDecision(status);
            foreach (var message in decision.Messages)
            {
                if (!string.IsNullOrWhiteSpace(message))
                {
                    AppendLog(message);
                }
            }

            if (!string.IsNullOrEmpty(decision.DeviceOverride))
            {
                args.Add("--device");
                args.Add(decision.DeviceOverride);
            }

            if (!string.IsNullOrEmpty(decision.ComputeTypeOverride))
            {
                args.Add("--compute_type");
                args.Add(decision.ComputeTypeOverride);
            }

            if (decision.AddSileroVad)
            {
                args.Add("--vad_method");
                args.Add("silero");
            }
        }

        AppendLog($"Using Whisper model '{SelectedModel}' for language '{languageForLog}'.");
        return args;
    }

    private AutoDecision BuildAutoDecision(TorchGpuStatus status)
    {
        var decision = new AutoDecision();

        if (status.HasError)
        {
            decision.Messages.Add("Could not auto-detect GPU status; defaulting to CPU-safe settings.");
            if (!string.IsNullOrWhiteSpace(status.ErrorMessage))
            {
                decision.Messages.Add("Details: " + status.ErrorMessage);
            }
            decision.DeviceOverride = "cpu";
            decision.ComputeTypeOverride = "float32";
            return decision;
        }

        if (!string.IsNullOrWhiteSpace(status.InfoMessage))
        {
            decision.Messages.Add(status.InfoMessage);
        }

        if (status.HasCuda && status.HasCudnn)
        {
            decision.Messages.Add("CUDA + cuDNN detected: using GPU acceleration with float16.");
        }
        else if (status.CudaAvailable == true && !status.HasCudnn)
        {
            decision.Messages.Add("CUDA detected but cuDNN missing (cudnn_ops_infer64_8.dll not found). Automatically forcing CPU mode.");
            decision.DeviceOverride = "cpu";
            decision.ComputeTypeOverride = "float32";
            decision.AddSileroVad = true;
            decision.Messages.Add("Also switching to Silero VAD to avoid pyannote model compatibility warnings.");
        }
        else
        {
            decision.Messages.Add("No CUDA detected: automatically adding --compute_type float32.");
            decision.ComputeTypeOverride = "float32";
        }

        return decision;
    }

    private async Task<TorchGpuStatus> ProbeTorchGpuStatusAsync(string pythonExe, CancellationToken token)
    {
        var tempScript = Path.Combine(Path.GetTempPath(), $"whisperx_gpu_probe_{Guid.NewGuid():N}.py");

        try
        {
            await File.WriteAllTextAsync(tempScript, TorchProbeScript, token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new TorchGpuStatus(null, null, null, $"Failed to write GPU probe script: {ex.Message}", null);
        }

        string stdout;
        string stderr;

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = GetWhisperXWorkingDirectory()
            };
            psi.ArgumentList.Add(tempScript);

            using var process = Process.Start(psi);
            if (process == null)
            {
                return new TorchGpuStatus(null, null, null, "Failed to launch Python for GPU probe.", null);
            }

            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync(token).ConfigureAwait(false);

            stdout = (await stdoutTask.ConfigureAwait(false)).Trim();
            stderr = (await stderrTask.ConfigureAwait(false)).Trim();
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new TorchGpuStatus(null, null, null, $"GPU probe execution failed: {ex.Message}", null);
        }
        finally
        {
            try { File.Delete(tempScript); } catch { }
        }

        if (string.IsNullOrWhiteSpace(stdout))
        {
            var message = string.IsNullOrWhiteSpace(stderr) ? "GPU probe produced no output." : stderr;
            return new TorchGpuStatus(null, null, null, message, null);
        }

        try
        {
            using var doc = JsonDocument.Parse(stdout);
            var root = doc.RootElement;

            if (root.TryGetProperty("error", out var errorProp) && errorProp.ValueKind == JsonValueKind.String)
            {
                var err = errorProp.GetString();
                var combined = string.IsNullOrWhiteSpace(stderr) ? err : $"{err} ({stderr})";
                return new TorchGpuStatus(null, null, null, combined, null);
            }

            bool cudaAvailable = root.TryGetProperty("cuda_available", out var cudaProp) && cudaProp.GetBoolean();
            int deviceCount = root.TryGetProperty("cuda_device_count", out var countProp) ? countProp.GetInt32() : 0;
            bool cudnnAvailable = root.TryGetProperty("cudnn_available", out var cudnnProp) && cudnnProp.GetBoolean();

            var info = string.IsNullOrWhiteSpace(stderr) ? null : stderr;

            return new TorchGpuStatus(cudaAvailable, deviceCount, cudnnAvailable, null, info);
        }
        catch (JsonException ex)
        {
            var message = $"Failed to parse GPU probe output: {ex.Message}. Raw output: {stdout}";
            var info = string.IsNullOrWhiteSpace(stderr) ? null : stderr;
            return new TorchGpuStatus(null, null, null, message, info);
        }
    }

    private static string RenderArgumentForLog(string argument)
    {
        if (string.IsNullOrEmpty(argument))
        {
            return "\"\"";
        }

        if (argument.Any(ch => char.IsWhiteSpace(ch) || ch == '"' || ch == '\''))
        {
            var escaped = argument.Replace("\"", "\\\"");
            return $"\"{escaped}\"";
        }

        return argument;
    }

    private string GetWhisperXWorkingDirectory()
    {
        // Return the project root directory where the .venv folder is located
        var currentDir = AppDomain.CurrentDomain.BaseDirectory;
        
        // Navigate up to find the project root (where .venv exists)
        var dir = new DirectoryInfo(currentDir);
        while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, ".venv")))
        {
            dir = dir.Parent;
        }
        
        return dir?.FullName ?? currentDir;
    }

    private static string? ResolvePython()
    {
        // Priority 1: Check for virtual environment Python in the project directory
        var currentDir = AppDomain.CurrentDomain.BaseDirectory;
        var dir = new DirectoryInfo(currentDir);
        
        // Navigate up to find the project root (where .venv exists)
        while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, ".venv")))
        {
            dir = dir.Parent;
        }
        
        if (dir != null)
        {
            var venvPython = Path.Combine(dir.FullName, ".venv", "Scripts", "python.exe");
            if (File.Exists(venvPython) && TryValidatePythonCandidate(venvPython, out var resolvedVenv))
            {
                return resolvedVenv;
            }
        }
        
        // Priority 2: Check WHISPERX_PYTHON environment variable
        var envPython = Environment.GetEnvironmentVariable("WHISPERX_PYTHON");
        if (!string.IsNullOrWhiteSpace(envPython) && TryValidatePythonCandidate(envPython, out var resolvedEnv))
        {
            return resolvedEnv;
        }

        // Priority 3: Standard Python candidates (as fallback)
        var localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

        var candidates = new List<string>
        {
            "python.exe",
            "python3.exe",
            "py.exe",
            "py"
        };

        if (!string.IsNullOrEmpty(localApp))
        {
            candidates.Add(Path.Combine(localApp, "Programs", "Python", "Python312", "python.exe"));
            candidates.Add(Path.Combine(localApp, "Programs", "Python", "Python311", "python.exe"));
            candidates.Add(Path.Combine(localApp, "Programs", "Python", "Python310", "python.exe"));
        }

        if (!string.IsNullOrEmpty(programFiles))
        {
            candidates.Add(Path.Combine(programFiles, "Python312", "python.exe"));
            candidates.Add(Path.Combine(programFiles, "Python311", "python.exe"));
            candidates.Add(Path.Combine(programFiles, "Python310", "python.exe"));
        }

        if (!string.IsNullOrEmpty(programFilesX86) && !string.Equals(programFiles, programFilesX86, StringComparison.OrdinalIgnoreCase))
        {
            candidates.Add(Path.Combine(programFilesX86, "Python312", "python.exe"));
            candidates.Add(Path.Combine(programFilesX86, "Python311", "python.exe"));
            candidates.Add(Path.Combine(programFilesX86, "Python310", "python.exe"));
        }

        foreach (var candidate in candidates)
        {
            if (TryValidatePythonCandidate(candidate, out var resolved))
            {
                return resolved;
            }
        }

        return null;
    }

    private static bool TryValidatePythonCandidate(string candidate, out string resolved)
    {
        resolved = string.Empty;
        if (string.IsNullOrWhiteSpace(candidate))
        {
            return false;
        }

        candidate = candidate.Trim().Trim('"');
        if (candidate.Length == 0)
        {
            return false;
        }

        var psi = new ProcessStartInfo
        {
            FileName = candidate,
            Arguments = "-c \"import sys; print(sys.version)\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        try
        {
            using var process = Process.Start(psi);
            if (process == null)
            {
                return false;
            }

            process.WaitForExit(5000); // 5 second timeout
            if (process.ExitCode != 0)
            {
                return false;
            }
            
            // If this is a virtual environment Python, also check if whisperx is available
            if (candidate.Contains(".venv") || candidate.Contains("venv"))
            {
                var whisperXCheckPsi = new ProcessStartInfo
                {
                    FileName = candidate,
                    Arguments = "-c \"import whisperx; print('WhisperX available')\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                
                using var whisperXProcess = Process.Start(whisperXCheckPsi);
                if (whisperXProcess == null)
                {
                    return false;
                }
                
                whisperXProcess.WaitForExit(5000);
                if (whisperXProcess.ExitCode != 0)
                {
                    // WhisperX not available in this environment
                    return false;
                }
            }

            resolved = Path.IsPathRooted(candidate)
                ? candidate
                : FindOnPath(candidate) ?? candidate;
            return true;
        }
        catch
        {
            // invalid candidate; ignore
            return false;
        }
    }

    private static string? FindOnPath(string exe)
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (path == null)
        {
            return null;
        }

        bool hasExtension = Path.HasExtension(exe);

        foreach (var segment in path.Split(Path.PathSeparator))
        {
            if (string.IsNullOrWhiteSpace(segment))
            {
                continue;
            }

            var candidate = Path.Combine(segment, exe);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            if (!hasExtension)
            {
                var exeCandidate = candidate + ".exe";
                if (File.Exists(exeCandidate))
                {
                    return exeCandidate;
                }
            }
        }

        return null;
    }

    private sealed class AutoDecision
    {
        public string? DeviceOverride { get; set; }
        public string? ComputeTypeOverride { get; set; }
        public bool AddSileroVad { get; set; }
        public List<string> Messages { get; } = new();
    }

    private readonly record struct TorchGpuStatus(bool? CudaAvailable, int? DeviceCount, bool? CudnnAvailable, string? ErrorMessage, string? InfoMessage)
    {
        public bool HasError => !string.IsNullOrWhiteSpace(ErrorMessage);
        public bool HasCuda => CudaAvailable == true && DeviceCount.GetValueOrDefault() > 0;
        public bool HasCudnn => CudnnAvailable == true;
    }

    private void AppendLog(string message, LogLevel level = LogLevel.Info)
    {
        Dispatcher.Invoke(() =>
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            
            // Parse and enhance common WhisperX messages
            var enhancedMessage = EnhanceMessage(message, ref level);
            
            // Create new paragraph for the log entry
            var paragraph = new Paragraph();
            paragraph.Margin = new Thickness(0, 1, 0, 1);
            
            // Timestamp
            var timeRun = new Run($"[{timestamp}] ") 
            { 
                Foreground = System.Windows.Media.Brushes.Gray,
                FontSize = 10
            };
            paragraph.Inlines.Add(timeRun);
            
            // Level indicator and message
            var levelRun = new Run(GetLevelPrefix(level) + " ");
            var messageRun = new Run(enhancedMessage);
            
            // Apply color based on level
            var brush = GetLevelBrush(level);
            messageRun.Foreground = brush;
            levelRun.Foreground = brush;
            
            paragraph.Inlines.Add(levelRun);
            paragraph.Inlines.Add(messageRun);
            
            // Add to RichTextBox
            LogDisplay.Document.Blocks.Add(paragraph);
            LogDisplay.ScrollToEnd();
            
            // Update progress if progress information detected
            UpdateProgressFromMessage(enhancedMessage);
        });
    }
    
    private string EnhanceMessage(string message, ref LogLevel level)
    {
        // Auto-detect message types and enhance formatting
        if (message.Contains(">>Performing"))
        {
            level = LogLevel.Info;
            if (message.Contains("voice activity detection"))
            {
                CurrentStep = "Voice Activity Detection";
                ProgressText = "Detecting speech segments...";
                return "🔊 Performing voice activity detection using VAD";
            }
            else if (message.Contains("transcription"))
            {
                CurrentStep = "Speech Transcription";
                ProgressText = "Converting speech to text...";
                return "🎤 Performing speech transcription using Whisper";
            }
            else if (message.Contains("alignment"))
            {
                CurrentStep = "Word-Level Alignment";
                ProgressText = "Aligning words with precise timestamps...";
                return "⏱️ Performing word-level alignment using wav2vec2";
            }
        }
        
        if (message.Contains("Auto-selected") || message.Contains("detected:"))
        {
            level = LogLevel.Success;
            return message;
        }
        
        if (message.Contains("Progress:"))
        {
            level = LogLevel.Progress;
            return message;
        }
        
        if (message.Contains("WARN:") || message.Contains("Warning"))
        {
            level = LogLevel.Warning;
            return message.Replace("WARN:", "").Replace("Warning:", "").Trim();
        }
        
        if (message.Contains("[ERR]") || message.Contains("ERROR:") || message.Contains("Error:"))
        {
            level = LogLevel.Error;
            return message.Replace("[ERR]", "").Replace("ERROR:", "").Replace("Error:", "").Trim();
        }
        
        if (message.Contains("Running:") || message.Contains("Command:"))
        {
            level = LogLevel.Debug;
            return message;
        }
        
        if (message.Contains("completed") || message.Contains("Done") || message.Contains("successfully"))
        {
            level = LogLevel.Success;
            return message;
        }
        
        if (message.Contains("model.safetensors:"))
        {
            level = LogLevel.Progress;
            var match = Regex.Match(message, @"(\d+)%.*?(\d+\.?\d*[KMGT]?B)\/(\d+\.?\d*[KMGT]?B)");
            if (match.Success)
            {
                return $"📥 Downloading model: {match.Groups[1].Value}% ({match.Groups[2].Value}/{match.Groups[3].Value})";
            }
        }
        
        if (message.Contains("Transcript:"))
        {
            level = LogLevel.Success;
            var transcriptMatch = Regex.Match(message, @"Transcript: \[([^\]]+)\] (.+)");
            if (transcriptMatch.Success)
            {
                var timeRange = transcriptMatch.Groups[1].Value;
                var text = transcriptMatch.Groups[2].Value;
                return $"📄 Transcript [{timeRange}]: {text.Substring(0, Math.Min(100, text.Length))}...";
            }
        }
        
        return message;
    }
    
    private string GetLevelPrefix(LogLevel level)
    {
        return level switch
        {
            LogLevel.Debug => "🔧",
            LogLevel.Info => "ℹ️",
            LogLevel.Success => "✅",
            LogLevel.Warning => "⚠️",
            LogLevel.Error => "❌",
            LogLevel.Progress => "📊",
            _ => "📝"
        };
    }
    
    private System.Windows.Media.Brush GetLevelBrush(LogLevel level)
    {
        return level switch
        {
            LogLevel.Debug => System.Windows.Media.Brushes.Gray,
            LogLevel.Info => new SolidColorBrush(System.Windows.Media.Color.FromRgb(46, 134, 193)),
            LogLevel.Success => new SolidColorBrush(System.Windows.Media.Color.FromRgb(40, 180, 99)),
            LogLevel.Warning => new SolidColorBrush(System.Windows.Media.Color.FromRgb(243, 156, 18)),
            LogLevel.Error => new SolidColorBrush(System.Windows.Media.Color.FromRgb(231, 76, 60)),
            LogLevel.Progress => new SolidColorBrush(System.Windows.Media.Color.FromRgb(142, 68, 173)),
            _ => System.Windows.Media.Brushes.Black
        };
    }
    
    private void UpdateProgressFromMessage(string message)
    {
        // Extract progress percentages from WhisperX output
        var progressMatch = Regex.Match(message, @"Progress:\s*(\d+\.?\d*)%");
        if (progressMatch.Success && double.TryParse(progressMatch.Groups[1].Value, out var progress))
        {
            ProgressValue = progress;
            IsIndeterminate = false;
            return;
        }
        
        // Handle model download progress
        var downloadMatch = Regex.Match(message, @"📥.*?(\d+)%");
        if (downloadMatch.Success && double.TryParse(downloadMatch.Groups[1].Value, out var downloadProgress))
        {
            ProgressValue = downloadProgress * 0.3; // Model download is roughly 30% of total
            ProgressText = "Downloading required models...";
            IsIndeterminate = false;
            return;
        }
        
        // Set indeterminate progress for processing steps
        if (message.Contains("🔊") || message.Contains("🎤") || message.Contains("⏱️"))
        {
            IsIndeterminate = true;
        }
    }
    
    private void UpdateFileInfo()
    {
        if (string.IsNullOrEmpty(InputFile) || !File.Exists(InputFile))
        {
            FileInfoText = "";
            ProcessingEstimate = "";
            return;
        }
        
        try
        {
            var fileInfo = new FileInfo(InputFile);
            var extension = fileInfo.Extension.ToLower();
            var isVideo = new[] { ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".3gp", ".mpg", ".mpeg" }
                .Contains(extension);
            
            FileInfoText = $"📁 File: {fileInfo.Name} | Size: {FormatFileSize(fileInfo.Length)} | Type: {(isVideo ? "Video" : "Audio")}";
            
            // Estimate processing time based on file size and selected model
            var estimatedMinutes = EstimateProcessingTime(fileInfo.Length, SelectedModel, isVideo);
            ProcessingEstimate = $"⏱️ Estimated processing time: ~{estimatedMinutes:F1} minutes with {SelectedModel} model";
        }
        catch (Exception ex)
        {
            FileInfoText = $"⚠️ Unable to read file information: {ex.Message}";
            ProcessingEstimate = "";
        }
    }
    
    private double EstimateProcessingTime(long fileSizeBytes, string model, bool isVideo)
    {
        // Base time estimates (minutes per MB for audio processing)
        var baseTimePerMB = model switch
        {
            "tiny" => 0.02,
            "base" => 0.03,
            "small" => 0.05,
            "medium" => 0.08,
            "large-v2" => 0.15,
            _ => 0.05
        };
        
        var fileSizeMB = fileSizeBytes / (1024.0 * 1024.0);
        var estimatedTime = fileSizeMB * baseTimePerMB;
        
        // Add overhead for video processing
        if (isVideo) estimatedTime *= 1.3;
        
        // Add overhead for diarization
        if (Diarize) estimatedTime *= 1.5;
        
        return Math.Max(0.5, estimatedTime); // Minimum 30 seconds
    }
    
    private void UpdateModelInfo()
    {
        var memoryReq = SelectedModel switch
        {
            "tiny" => "~1GB",
            "base" => "~2GB", 
            "small" => "~4GB",
            "medium" => "~8GB",
            "large-v2" => "~16GB",
            _ => "~4GB"
        };
        
        var accuracy = SelectedModel switch
        {
            "tiny" => "Basic",
            "base" => "Good",
            "small" => "Very Good", 
            "medium" => "Excellent",
            "large-v2" => "Outstanding",
            _ => "Very Good"
        };
        
        ModelInfo = $"🧠 Model: {SelectedModel} | Memory: {memoryReq} | Accuracy: {accuracy}";
    }
    
    private string FormatFileSize(long bytes)
    {
        string[] sizes = { "B", "KB", "MB", "GB" };
        double len = bytes;
        int order = 0;
        while (len >= 1024 && order < sizes.Length - 1)
        {
            order++;
            len = len / 1024;
        }
        return $"{len:0.##} {sizes[order]}";
    }
}
