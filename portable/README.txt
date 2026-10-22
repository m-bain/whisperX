WhisperX Windows Portable (CUDA 12.8)
===================================
No local Python or pip installation is needed.

1. Extract all release .7z.00* volumes together using 7-Zip.
2. Drag an audio file onto run.bat (or open run.bat and paste its path).
3. Output files are written to the output directory.
4. On the first run, WhisperX downloads model files to models/.

NVIDIA GPU use requires a compatible NVIDIA display driver. Building on
GitHub Actions does not validate GPU operation because hosted Windows runners
do not have NVIDIA GPUs. CPU operation can be requested by invoking:
  run.bat "C:\path\to\audio.mp3" --device cpu --compute_type int8

Speaker diarization requires Hugging Face access to the gated pyannote model,
acceptance of its model terms, and passing --diarize --hf_token YOUR_TOKEN.
Do not commit or share personal tokens.

Large release files may arrive as multiple .7z.001, .7z.002 ... pieces.
All pieces are required. This is not a single ZIP file.
