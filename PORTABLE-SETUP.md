# Windows Portable build (CUDA 12.8)

This repository includes a GitHub Actions workflow at
`.github/workflows/windows-portable.yml` for a Windows x64 portable build
containing managed Python 3.11, WhisperX dependencies and FFmpeg.

## Test the build

Open **Actions → Windows Portable (CUDA 12.8) → Run workflow**, choosing
the branch containing the workflow. Download the Actions artifact after a
successful build. Note: CI tests imports and CLI help, not GPU inference.

## Publish

After merging the workflow to the default branch, push a tag starting with
`portable-v`, for example `portable-v1.0.0`, to create a GitHub Release.

## Use the portable distribution

GitHub Releases limit individual assets to 2 GiB. The workflow produces
multipart 7z volumes under that limit. Download every numbered part, use
7-Zip to extract the first one, then run `WhisperX/run.bat`.
No pip or locally installed Python is needed. Models download on first use.

## Known limitations

- GPU inference requires an NVIDIA driver compatible with CUDA 12.8 and
  local testing on an NVIDIA GPU.
- The GitHub-hosted Windows runner cannot confirm GPU inference works.
- The bundle is large because PyTorch's CUDA stack is included.
- Online access is required to download models not already cached.
- Diarization needs a Hugging Face access token and model access acceptance.
- This workflow deliberately creates multipart 7z archives, not a ZIP,
  to support large files.
