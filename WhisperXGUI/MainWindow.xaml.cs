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
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using WinForms = System.Windows.Forms;

namespace WhisperXGUI;

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

    public string InputFile { get => _inputFile; set { _inputFile = value; OnPropertyChanged(); UpdateCanRun(); } }
    public string OutputFolder { get => _outputFolder; set { _outputFolder = value; OnPropertyChanged(); UpdateCanRun(); } }
    public string LogText { get => _logText; set { _logText = value; OnPropertyChanged(); } }
    public string Status { get => _status; set { _status = value; OnPropertyChanged(); } }
    public bool IsRunning { get => _isRunning; set { _isRunning = value; OnPropertyChanged(); OnPropertyChanged(nameof(CanRun)); } }
    public string SelectedModel { get => _selectedModel; set { _selectedModel = value; OnPropertyChanged(); } }
    public string SelectedFormat { get => _selectedFormat; set { _selectedFormat = value; OnPropertyChanged(); } }
    public string SelectedDevice { get => _selectedDevice; set { _selectedDevice = value; OnPropertyChanged(); } }
    public bool Diarize { get => _diarize; set { _diarize = value; OnPropertyChanged(); } }
    public string LanguageOverride { get => _languageOverride; set { _languageOverride = value; OnPropertyChanged(); } }

    public bool CanRun => !IsRunning && File.Exists(InputFile) && Directory.Exists(OutputFolder);

    public event PropertyChangedEventHandler? PropertyChanged;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;
        OutputFolder = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
        Loaded += MainWindow_Loaded;
    }

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        Loaded -= MainWindow_Loaded;
        await DetectCudaSupportAsync();
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

    private void UpdateCanRun() => OnPropertyChanged(nameof(CanRun));

    private void BrowseFile_Click(object sender, RoutedEventArgs e)
    {
        var ofd = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Media Files|*.mp4;*.mkv;*.wav;*.mp3;*.flac;*.ogg;*.mov|All Files|*.*",
            CheckFileExists = true
        };

        if (ofd.ShowDialog() == true)
        {
            InputFile = ofd.FileName;

            var parent = Path.GetDirectoryName(InputFile);
            if ((string.IsNullOrWhiteSpace(OutputFolder) || !Directory.Exists(OutputFolder)) && !string.IsNullOrEmpty(parent))
            {
                OutputFolder = parent;
            }
        }
    }

    private void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        using var fbd = new WinForms.FolderBrowserDialog();
        if (fbd.ShowDialog() == WinForms.DialogResult.OK && !string.IsNullOrEmpty(fbd.SelectedPath))
        {
            OutputFolder = fbd.SelectedPath;
        }
    }

    private async void Run_Click(object sender, RoutedEventArgs e)
    {
        if (!CanRun) return;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        Status = "Running...";
        LogText = string.Empty;

        try
        {
            await RunWhisperXAsync(_cts.Token);
            Status = "Completed";
        }
        catch (OperationCanceledException)
        {
            Status = "Canceled";
            AppendLog("Operation canceled by user.");
        }
        catch (Exception ex)
        {
            Status = "Error";
            AppendLog("ERROR: " + ex.Message);
        }
        finally
        {
            IsRunning = false;
        }
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => _cts?.Cancel();

    private async Task RunWhisperXAsync(CancellationToken token)
    {
        var pythonExe = ResolvePython();
        if (string.IsNullOrEmpty(pythonExe))
            throw new InvalidOperationException("Python executable not found or not usable. Install Python 3.10+ or set WHISPERX_PYTHON.");

        var argumentList = await BuildWhisperArgumentListAsync(pythonExe, token).ConfigureAwait(false);

        AppendLog($"> Running: {pythonExe} {string.Join(' ', argumentList.Select(RenderArgumentForLog))}");

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

        process.OutputDataReceived += (_, e) => { if (e.Data != null) AppendLog(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data != null) AppendLog("[ERR] " + e.Data); };
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

        AppendLog("Done.");
    }

    private async Task DetectCudaSupportAsync()
    {
        var pythonExe = ResolvePython();
        if (string.IsNullOrEmpty(pythonExe))
        {
            AppendLog("WARN: Unable to probe CUDA support because Python is not installed or not accessible.");
            return;
        }

        AppendLog($"> Probing CUDA/cuDNN support via {pythonExe}");
        try
        {
            var status = await ProbeTorchGpuStatusAsync(pythonExe, CancellationToken.None);
            _cachedGpuStatus = status;

            if (status.HasError)
            {
                AppendLog("WARN: Could not auto-detect GPU status; defaulting to CPU-safe settings.");
                if (!string.IsNullOrWhiteSpace(status.ErrorMessage))
                {
                    AppendLog("WARN: " + status.ErrorMessage);
                }
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                return;
            }

            if (!string.IsNullOrWhiteSpace(status.InfoMessage))
            {
                AppendLog(status.InfoMessage);
            }

            if (status.HasCuda && status.HasCudnn)
            {
                if (!Devices.Contains("cuda"))
                {
                    Devices.Add("cuda");
                }
                AppendLog("CUDA + cuDNN detected: GPU acceleration is available.");
            }
            else if (status.CudaAvailable == true && !status.HasCudnn)
            {
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                AppendLog("CUDA detected but cuDNN missing (cudnn_ops_infer64_8.dll not found). WhisperX will default to CPU unless cuDNN is installed.");
            }
            else
            {
                Devices.Remove("cuda");
                if (SelectedDevice == "cuda")
                {
                    SelectedDevice = "auto";
                }
                AppendLog("CUDA not available; CPU mode will be used by default.");
            }
        }
        catch (Exception ex)
        {
            AppendLog("WARN: Failed to probe CUDA support: " + ex.Message);
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
        => AppDomain.CurrentDomain.BaseDirectory;

    private static string? ResolvePython()
    {
        var localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);

        var candidates = new List<string>
        {
            (Environment.GetEnvironmentVariable("WHISPERX_PYTHON") ?? string.Empty).Trim().Trim('"'),
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
            Arguments = "-c \"import sys\"",
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

            process.WaitForExit();
            if (process.ExitCode == 0)
            {
                resolved = Path.IsPathRooted(candidate)
                    ? candidate
                    : FindOnPath(candidate) ?? candidate;
                return true;
            }
        }
        catch
        {
            // invalid candidate; ignore
        }

        return false;
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

    private void AppendLog(string line)
    {
        Dispatcher.Invoke(() =>
        {
            LogText += line + Environment.NewLine;
        });
    }
}
