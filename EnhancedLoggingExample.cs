// Enhanced logging implementation for WhisperX GUI
// This shows how to implement the key improvements identified in the analysis

using System;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Documents;
using System.Windows.Media;

namespace WhisperXGUI
{
    public enum LogLevel
    {
        Debug,
        Info,
        Success, 
        Warning,
        Error,
        Progress
    }

    public partial class MainWindow
    {
        private DateTime _startTime;
        private string _currentStep = "Ready";
        private double _progressValue = 0;
        private bool _isIndeterminate = false;
        
        // Enhanced properties for better status tracking
        public string CurrentStep 
        { 
            get => _currentStep; 
            set { _currentStep = value; OnPropertyChanged(); } 
        }
        
        public double ProgressValue 
        { 
            get => _progressValue; 
            set { _progressValue = value; OnPropertyChanged(); OnPropertyChanged(nameof(ProgressPercent)); } 
        }
        
        public string ProgressPercent => $"{ProgressValue:F1}%";
        
        public string ElapsedTime
        {
            get
            {
                if (_startTime == default) return "00:00:00";
                var elapsed = DateTime.Now - _startTime;
                return elapsed.ToString(@"hh\:mm\:ss");
            }
        }

        // Enhanced AppendLog with message categorization and formatting
        private void AppendLog(string message, LogLevel level = LogLevel.Info)
        {
            Dispatcher.Invoke(() =>
            {
                var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
                
                // Parse and enhance common WhisperX messages
                var enhancedMessage = EnhanceMessage(message, ref level);
                
                // Create formatted log entry
                var logEntry = FormatLogEntry(timestamp, enhancedMessage, level);
                
                // Add to log display
                LogText += logEntry + Environment.NewLine;
                
                // Update progress if progress information detected
                UpdateProgressFromMessage(enhancedMessage);
                
                // Trigger property change to update UI
                OnPropertyChanged(nameof(LogText));
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
                    return "🔊 Detecting speech segments using VAD...";
                }
                else if (message.Contains("transcription"))
                {
                    CurrentStep = "Speech Transcription";
                    return "🎤 Converting speech to text using Whisper...";
                }
                else if (message.Contains("alignment"))
                {
                    CurrentStep = "Word-Level Alignment";
                    return "⏱️ Aligning words with precise timestamps...";
                }
            }
            
            if (message.Contains("Auto-selected") || message.Contains("detected:"))
            {
                level = LogLevel.Success;
                return "✅ " + message;
            }
            
            if (message.Contains("Progress:"))
            {
                level = LogLevel.Progress;
                return "📊 " + message;
            }
            
            if (message.Contains("WARN:") || message.Contains("Warning"))
            {
                level = LogLevel.Warning;
                return "⚠️ " + message.Replace("WARN:", "").Replace("Warning:", "").Trim();
            }
            
            if (message.Contains("[ERR]") || message.Contains("ERROR:") || message.Contains("Error:"))
            {
                level = LogLevel.Error;
                return "❌ " + message.Replace("[ERR]", "").Replace("ERROR:", "").Replace("Error:", "").Trim();
            }
            
            if (message.Contains("Running:") || message.Contains("Command:"))
            {
                level = LogLevel.Debug;
                return "🔧 " + message;
            }
            
            if (message.Contains("completed") || message.Contains("Done") || message.Contains("successfully"))
            {
                level = LogLevel.Success;
                return "✅ " + message;
            }
            
            // Default enhancement
            return "📝 " + message;
        }
        
        private string FormatLogEntry(string timestamp, string message, LogLevel level)
        {
            var prefix = level switch
            {
                LogLevel.Debug => "[DEBUG]",
                LogLevel.Info => "[INFO] ",
                LogLevel.Success => "[DONE] ",
                LogLevel.Warning => "[WARN] ",
                LogLevel.Error => "[ERROR]",
                LogLevel.Progress => "[PROG] ",
                _ => "[LOG]  "
            };
            
            return $"[{timestamp}] {prefix} {message}";
        }
        
        private void UpdateProgressFromMessage(string message)
        {
            // Extract progress percentages from WhisperX output
            var progressMatch = Regex.Match(message, @"Progress:\s*(\d+\.?\d*)%");
            if (progressMatch.Success && double.TryParse(progressMatch.Groups[1].Value, out var progress))
            {
                ProgressValue = progress;
                _isIndeterminate = false;
                OnPropertyChanged(nameof(IsIndeterminate));
                return;
            }
            
            // Set indeterminate progress for processing steps
            if (message.Contains("🔊") || message.Contains("🎤") || message.Contains("⏱️"))
            {
                _isIndeterminate = true;
                OnPropertyChanged(nameof(IsIndeterminate));
            }
        }
        
        // Enhanced process output handling
        private void HandleProcessOutput(string output)
        {
            if (string.IsNullOrWhiteSpace(output)) return;
            
            // Handle special WhisperX output patterns
            if (output.Contains("model.safetensors:") && output.Contains("%"))
            {
                // Model download progress
                var match = Regex.Match(output, @"(\d+)%.*?(\d+\.?\d*[KMGT]?B)\/(\d+\.?\d*[KMGT]?B)");
                if (match.Success)
                {
                    AppendLog($"📥 Downloading model: {match.Groups[1].Value}% ({match.Groups[2].Value}/{match.Groups[3].Value})", LogLevel.Progress);
                    return;
                }
            }
            
            if (output.Contains("Lightning automatically upgraded"))
            {
                AppendLog("⚡ Model compatibility: Upgrading checkpoint format", LogLevel.Info);
                return;
            }
            
            if (output.Contains("Model was trained with"))
            {
                AppendLog("⚠️ Model version compatibility warning (non-critical)", LogLevel.Warning);
                return;
            }
            
            // Handle transcript output
            if (output.Contains("Transcript:"))
            {
                var transcriptMatch = Regex.Match(output, @"Transcript: \[([^\]]+)\] (.+)");
                if (transcriptMatch.Success)
                {
                    var timeRange = transcriptMatch.Groups[1].Value;
                    var text = transcriptMatch.Groups[2].Value;
                    AppendLog($"📄 Transcript [{timeRange}]: {text.Substring(0, Math.Min(100, text.Length))}...", LogLevel.Success);
                    return;
                }
            }
            
            // Default handling
            AppendLog(output);
        }
        
        // Enhanced run process with better status tracking
        private async Task RunWhisperXAsync(CancellationToken token)
        {
            _startTime = DateTime.Now;
            var timer = new System.Timers.Timer(1000); // Update elapsed time every second
            timer.Elapsed += (s, e) => OnPropertyChanged(nameof(ElapsedTime));
            timer.Start();
            
            try
            {
                AppendLog("🚀 Starting WhisperX processing...", LogLevel.Info);
                
                var pythonExe = ResolvePython();
                if (string.IsNullOrEmpty(pythonExe))
                    throw new InvalidOperationException("Python executable not found");
                
                AppendLog($"🐍 Using Python: {pythonExe}", LogLevel.Debug);
                
                // Log file information
                var fileInfo = new System.IO.FileInfo(InputFile);
                AppendLog($"📁 Input file: {fileInfo.Name} ({FormatFileSize(fileInfo.Length)})", LogLevel.Info);
                
                var argumentList = await BuildWhisperArgumentListAsync(pythonExe, token);
                AppendLog($"🔧 Command: {string.Join(' ', argumentList.Select(RenderArgumentForLog))}", LogLevel.Debug);
                
                // Start process with enhanced output handling
                var startInfo = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding = System.Text.Encoding.UTF8,
                    WorkingDirectory = GetWhisperXWorkingDirectory()
                };
                
                foreach (var arg in argumentList)
                {
                    startInfo.ArgumentList.Add(arg);
                }
                
                using var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
                var tcs = new TaskCompletionSource<int>();
                
                // Enhanced output handling
                process.OutputDataReceived += (_, e) => {
                    if (e.Data != null) HandleProcessOutput(e.Data);
                };
                process.ErrorDataReceived += (_, e) => {
                    if (e.Data != null) HandleProcessOutput(e.Data);
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
                            AppendLog("🛑 Cancelling process...", LogLevel.Warning);
                            process.Kill(entireProcessTree: true);
                        }
                    }
                    catch { }
                }))
                {
                    int exitCode = await tcs.Task;
                    if (exitCode != 0)
                        throw new InvalidOperationException($"WhisperX exited with code {exitCode}");
                }
                
                var elapsed = DateTime.Now - _startTime;
                AppendLog($"🎉 Processing completed in {elapsed:mm\\:ss}", LogLevel.Success);
                CurrentStep = "Completed";
                ProgressValue = 100;
                
            }
            finally
            {
                timer?.Stop();
                timer?.Dispose();
            }
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
}