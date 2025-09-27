# WhisperX Windows Usage Examples

## 🎯 Real-World Scenarios

### Meeting & Conference Recording
```powershell
# Corporate meeting with multiple speakers
.\run_whisperx.ps1 quarterly_meeting.mp4 --diarize --max_speakers 5 --language en --output_format all --output_dir "meetings\Q4"

# Chinese business presentation
.\run_whisperx.ps1 business_presentation.mp4 --language zh --output_format srt --model medium --output_dir "presentations"
```

### Educational Content
```powershell
# University lecture recording
.\run_whisperx.ps1 physics_lecture.mp4 --language en --model large-v2 --output_format vtt --highlight_words True

# Language learning content
.\run_whisperx.ps1 chinese_lesson.mp4 --language zh --output_format all --model medium
```

### Content Creation & Media
```powershell
# YouTube video subtitles
.\run_whisperx.ps1 youtube_video.mp4 --language en --output_format srt --model medium --highlight_words True

# Podcast transcription
.\run_whisperx.ps1 podcast_episode.mp3 --diarize --max_speakers 3 --language en --output_format txt,json
```

## 🔧 Performance Optimization Examples

### Speed vs Quality Trade-offs
```powershell
# Maximum speed (draft quality)
.\run_whisperx.ps1 long_video.mp4 --model tiny --no_align --language en --output_format txt

# Balanced speed/quality (recommended)
.\run_whisperx.ps1 video.mp4 --model small --language en --output_format srt

# Maximum quality (professional use)  
.\run_whisperx.ps1 important_video.mp4 --model large-v2 --diarize --language en --output_format all
```

### Hardware-Optimized Processing
```powershell
# High-end GPU (RTX 4090/3080)
.\run_whisperx.ps1 video.mp4 --model large-v2 --compute_type float16 --batch_size 16 --language en

# Mid-range GPU (RTX 3060/1660)  
.\run_whisperx.ps1 video.mp4 --model medium --compute_type float16 --batch_size 8 --language en

# CPU-only optimized
.\run_whisperx.ps1 video.mp4 --model small --device cpu --compute_type int8 --threads 4 --language en
```

## 🌍 Multilingual Processing

### Common Languages
```powershell
# English content
.\run_whisperx.ps1 english_video.mp4 --language en --output_format srt

# Chinese (auto-detects Simplified/Traditional)
.\run_whisperx.ps1 chinese_video.mp4 --language zh --output_format srt  

# Japanese with diarization
.\run_whisperx.ps1 japanese_meeting.mp4 --language ja --diarize --output_format vtt

# Spanish presentation  
.\run_whisperx.ps1 spanish_lecture.mp4 --language es --model medium --output_format all

# French interview
.\run_whisperx.ps1 french_interview.mp4 --language fr --diarize --max_speakers 2 --output_format srt
```

### Auto-Language Detection
```powershell
# Unknown language (slower but flexible)
.\run_whisperx.ps1 unknown_content.mp4 --output_format srt

# Mixed language content
.\run_whisperx.ps1 multilingual_conference.mp4 --output_format json  # JSON includes detected language
```

## 📁 Batch Processing Examples

### Process Video Library
```powershell
# All MP4 files with English subtitles
Get-ChildItem "C:\Videos\*.mp4" | ForEach-Object {
    $outputName = $_.BaseName + ".srt"
    .\run_whisperx.ps1 $_.FullName --language en --output_format srt --output_dir "C:\Subtitles"
}

# Organize by language
$videos = @(
    @{ path="C:\English\*.mp4"; lang="en" },
    @{ path="C:\Chinese\*.mp4"; lang="zh" },  
    @{ path="C:\Japanese\*.mp4"; lang="ja" }
)

foreach ($videoSet in $videos) {
    Get-ChildItem $videoSet.path | ForEach-Object {
        .\run_whisperx.ps1 $_.FullName --language $videoSet.lang --output_format srt --output_dir "C:\Subtitles\$($videoSet.lang)"
    }
}
```

### Parallel Processing (Advanced)
```powershell
# Process multiple videos simultaneously (CPU cores permitting)
$videos = Get-ChildItem "*.mp4"
$videos | ForEach-Object -Parallel {
    & "D:\whisperX_for_Windows\run_whisperx.ps1" $_.FullName --language en --output_format srt --output_dir "output"
} -ThrottleLimit 3  # Limit concurrent jobs based on available RAM
```

## 🎥 Video Format Examples

### Common Input Formats
```powershell
# MP4 video
.\run_whisperx.ps1 movie.mp4 --language en --output_format srt

# MKV high-quality video  
.\run_whisperx.ps1 documentary.mkv --language en --model large-v2 --output_format vtt

# AVI legacy format
.\run_whisperx.ps1 old_recording.avi --language en --output_format srt --model medium

# MOV from iPhone/Mac
.\run_whisperx.ps1 phone_video.mov --language en --output_format srt

# WebM from web downloads
.\run_whisperx.ps1 webinar.webm --language en --diarize --output_format all
```

### Audio-Only Processing
```powershell
# WAV audio file
.\run_whisperx.ps1 recording.wav --language en --output_format json,txt

# MP3 podcast
.\run_whisperx.ps1 podcast.mp3 --diarize --language en --output_format srt,txt

# FLAC high-quality audio
.\run_whisperx.ps1 interview.flac --language en --model large-v2 --output_format all
```

## 🔊 Audio Quality & Processing

### Handling Different Audio Quality
```powershell
# Clean studio recording (can use larger model)
.\run_whisperx.ps1 studio_recording.wav --model large-v2 --language en --output_format srt

# Noisy recording (use robust settings)
.\run_whisperx.ps1 noisy_meeting.mp4 --vad_method silero --model medium --language en --output_format srt

# Phone call recording (optimize for speech)
.\run_whisperx.ps1 phone_call.wav --model small --vad_method silero --language en --output_format txt
```

### Speaker Separation Examples
```powershell
# Interview (2 speakers)
.\run_whisperx.ps1 interview.mp4 --diarize --min_speakers 2 --max_speakers 2 --language en --output_format srt

# Panel discussion (up to 6 speakers)
.\run_whisperx.ps1 panel.mp4 --diarize --max_speakers 6 --language en --output_format vtt

# Large meeting (unknown speaker count)
.\run_whisperx.ps1 town_hall.mp4 --diarize --language en --output_format json  # JSON shows detected speakers
```

## 📊 Output Format Comparisons

### SRT (SubRip) - Most Compatible
```powershell
.\run_whisperx.ps1 video.mp4 --language en --output_format srt
```
**Output:**
```
1
00:00:00,000 --> 00:00:05,120
Hello and welcome to today's presentation.

2  
00:00:05,120 --> 00:00:08,960
We'll be covering several important topics.
```

### VTT (WebVTT) - Web Optimized
```powershell  
.\run_whisperx.ps1 video.mp4 --language en --output_format vtt --highlight_words True
```
**Output:**
```
WEBVTT

00:00:00.000 --> 00:00:05.120
Hello and welcome to today's presentation.

00:00:05.120 --> 00:00:08.960
We'll be covering several important topics.
```

### JSON - Programming/Analysis
```powershell
.\run_whisperx.ps1 video.mp4 --language en --output_format json --diarize
```
**Output:** Structured data with word-level timing, confidence scores, and speaker labels.

### TXT - Plain Text
```powershell
.\run_whisperx.ps1 video.mp4 --language en --output_format txt
```
**Output:** Clean text transcript without timing information.

## 🛠️ Troubleshooting Examples

### When Processing Fails
```powershell
# Retry with maximum compatibility
.\run_whisperx.ps1 problematic_video.mp4 --device cpu --compute_type float32 --vad_method silero --model base

# Skip alignment if word-level timing fails  
.\run_whisperx.ps1 video.mp4 --no_align --language en --output_format srt

# Use tiny model for very limited resources
.\run_whisperx.ps1 video.mp4 --model tiny --device cpu --compute_type int8 --language en
```

### Large File Processing
```powershell
# Long video (2+ hours) - use efficient settings
.\run_whisperx.ps1 long_lecture.mp4 --model medium --chunk_size 20 --language en --output_format srt

# Very large file - preprocess with FFmpeg first
ffmpeg -i huge_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio_only.wav
.\run_whisperx.ps1 audio_only.wav --language en --output_format srt
```

## 📈 Quality Benchmarks

### Model Comparison (1-hour English lecture)
| Model | Processing Time | CPU Usage | RAM Usage | Quality Score |
|-------|-----------------|-----------|-----------|---------------|
| tiny | 5 minutes | Low | 1GB | 3/5 ⭐ |
| base | 8 minutes | Medium | 2GB | 4/5 ⭐ |
| small | 12 minutes | Medium | 4GB | 4.5/5 ⭐ |
| medium | 20 minutes | High | 8GB | 4.8/5 ⭐ |
| large-v2 | 35 minutes | High | 16GB | 5/5 ⭐ |

### Language-Specific Recommendations
| Language | Recommended Model | Notes |
|----------|------------------|-------|
| English | medium/large-v2 | Best alignment model support |
| Chinese | medium/large-v2 | Good for both Simplified/Traditional |  
| Japanese | large-v2 | Complex writing system needs larger model |
| Spanish | medium | Good balance for Romance languages |
| French | medium | Excellent native support |
| German | medium/large-v2 | Compound words benefit from larger model |

## 💼 Professional Workflows

### Corporate Meeting Processing
```powershell
# Standard corporate workflow
$meetingFiles = Get-ChildItem "meetings\*.mp4"
foreach ($meeting in $meetingFiles) {
    $date = $meeting.Name.Split('_')[0]  # Assuming YYYY-MM-DD_meeting.mp4 format
    $outputDir = "transcripts\$date"
    
    .\run_whisperx.ps1 $meeting.FullName --diarize --max_speakers 8 --language en --output_format all --output_dir $outputDir
    Write-Host "Processed meeting for $date" -ForegroundColor Green
}
```

### Content Creator Pipeline  
```powershell
# YouTube content workflow
$videos = Get-ChildItem "content\*.mp4"
foreach ($video in $videos) {
    # Generate multiple subtitle formats for different platforms
    .\run_whisperx.ps1 $video.FullName --language en --output_format srt,vtt --highlight_words True --output_dir "subtitles"
    
    # Create plain text for description/blog posts
    .\run_whisperx.ps1 $video.FullName --language en --output_format txt --output_dir "descriptions"
}
```

### Academic Research Processing
```powershell
# Research interview processing
$interviews = Get-ChildItem "research\interviews\*.mp3"
foreach ($interview in $interviews) {
    # High-accuracy transcription with speaker identification
    .\run_whisperx.ps1 $interview.FullName --model large-v2 --diarize --language en --output_format json,txt --output_dir "research\transcripts"
    
    # Generate summary-friendly format
    .\run_whisperx.ps1 $interview.FullName --no_align --model medium --language en --output_format txt --output_dir "research\summaries"
}
```

These examples demonstrate the full range of WhisperX capabilities on Windows, from basic transcription to complex professional workflows.