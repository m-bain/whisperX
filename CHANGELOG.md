# Changelog - Docker Deployment Branch

## Overview

Added complete Docker deployment support with web UI for easy self-hosting. No more messing with Python environments - just pull and run!

## What's New

### Core Files Added (in `docker/` directory)

- **`Dockerfile`** - GPU-enabled image (NVIDIA CUDA 12.1.1, ~14.7GB)
- **`Dockerfile.cpu`** - CPU-only image (Python 3.11 slim, ~3.8GB) with CTranslate2 fix
- **`docker-compose.yml`** - Easy docker-compose setup with privileged mode
- **`api.py`** - FastAPI backend with model management
- **`web/index.html`** - Modern drag-and-drop web interface
- **`requirements-api.txt`** - API dependencies
- **`entrypoint.sh`** - Fixes CTranslate2 executable stack issue with setarch -R
- **`.dockerignore`** - Optimized build context

### Documentation

- **`DOCKER_DEPLOYMENT.md`** - Complete deployment guide (homelab-friendly!)

### Features

**Web Interface:**
- 🌐 Drag-and-drop file upload
- 📊 Real-time progress tracking with visual feedback
- 🤖 Model management panel (download models on-demand)
- 📥 Shows available downloaded models
- 🔐 Hugging Face token integration for speaker diarization
- 📝 Multiple export formats (JSON, SRT, VTT, TXT, TSV)
- 🎨 Clean, modern UI that actually looks good

**API Enhancements:**
- `/models/list` - List currently downloaded models
- `/models/download` - Download models with progress tracking
- `/models/download/progress/{task_id}` - Real-time download progress
- `/transcribe` - Background job processing with progress updates
- `/health` - Health check endpoint
- `/docs` - Auto-generated Swagger documentation

**Deployment:**
- 🐳 Multi-platform Docker images (linux/amd64, linux/arm64 for CPU)
- 🔄 Automated GitHub Actions CI/CD
- 📦 Published to Docker Hub
- 🏷️ Proper version tagging
- ⚡ Build caching for faster iterations
- 🧪 Automated testing in CI

### Optimizations

**Image Size Reductions:**
- CPU: From 14GB → 3.8GB (73% smaller!)
- GPU: From 27.8GB → 14.7GB (47% smaller!)

**How We Got There:**
- ✅ Runtime-only base images (no dev tools)
- ✅ No pre-installed models (download on-demand)
- ✅ Optimized layer caching
- ✅ Multi-stage builds where applicable
- ✅ Minimal system dependencies

**Note:** GPU image is 14.7GB because:
- CUDA runtime: ~2.2GB (needed for GPU)
- PyTorch with CUDA: ~6-7GB (GPU kernels are huge)
- ML libraries: ~3-4GB
- This is normal for production GPU ML images

## Changes to Existing Files

### Modified

- Updated Python dependency versions for compatibility
- Minor tweaks to support Docker environment

### No Breaking Changes

- All original CLI functionality preserved
- Existing Python API unchanged
- Original usage still works exactly the same

## Technical Details

### Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (localhost)    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   FastAPI       │
│   (api.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   WhisperX      │
│   Library       │
└─────────────────┘
```

### CI/CD Pipeline

```
Push/Tag → GitHub Actions → Build Images → Test → Push to Docker Hub → Done
```

**Build Matrix:**
- CPU image: linux/amd64 + linux/arm64
- GPU image: linux/amd64 only (CUDA limitation)

**Tagging Strategy:**
- `latest` → GPU version (most common use case)
- `cpu-latest` → CPU version
- `gpu-latest` → GPU version
- `v1.0.0` → Versioned releases
- Branch names → `branch-name-cpu`, `branch-name-gpu`

## Quick Start Examples

### For the Impatient

```bash
# CPU (works anywhere)
docker run -d -p 8000:8000 your-username/whisperx:cpu-latest

# GPU (needs NVIDIA GPU)
docker run -d --gpus all -p 8000:8000 your-username/whisperx:gpu-latest
```

Open http://localhost:8000 - Done!

### For the Docker Compose Fans

```yaml
version: '3.8'
services:
  whisperx:
    image: your-username/whisperx:cpu-latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

Run: `docker-compose up -d`

### For the Homelab Heroes

With persistent storage and all the bells:

```bash
docker run -d \
  --name whisperx \
  -p 8000:8000 \
  -v ~/whisperx/models:/app/models \
  -v ~/whisperx/input:/input \
  -v ~/whisperx/output:/output \
  -e HF_TOKEN=your_hf_token \
  --restart unless-stopped \
  your-username/whisperx:cpu-latest
```

## Deployment Workflow

1. **Setup** (5 min)
   - Create Docker Hub account
   - Generate access token
   - Add GitHub secrets

2. **Deploy** (1 min)
   - Push code or create tag
   - GitHub Actions handles the rest

3. **Use** (30 min later)
   - Images built and published
   - Pull and run anywhere

## Performance Notes

**Model Download Speeds:**
- tiny (39MB): ~5 seconds
- base (142MB): ~15 seconds
- medium (1.42GB): ~2 minutes
- large-v3 (2.87GB): ~4 minutes

*Depends on internet connection*

**Transcription Speed (1 min audio):**
- tiny: CPU ~30s, GPU ~5s
- base: CPU ~1min, GPU ~8s
- medium: CPU ~8min, GPU ~20s

## Known Limitations

- GPU image only supports linux/amd64 (CUDA limitation)
- Large image sizes for GPU (inherent to CUDA + PyTorch)
- First model download takes time (cached after)
- Requires Docker 20.10+ for GPU support

## Future Ideas

Things that could be added:
- [ ] Authentication layer (OAuth2?)
- [ ] Batch processing queue
- [ ] Webhook notifications
- [ ] S3/cloud storage integration
- [ ] Prometheus metrics
- [ ] Multi-user support
- [ ] API rate limiting

## Credits

Built on top of:
- **WhisperX** by Max Bain
- **Faster Whisper** by Guillaume Klein
- **PyAnnote Audio** by Hervé Bredin
- **FastAPI** by Sebastián Ramírez

## License

Same as WhisperX - BSD 4-Clause License

---

## How to Use This Branch

### Testing Locally

```bash
git clone https://github.com/m-bain/whisperX.git
cd whisperX
git checkout docker-image-web-interface

# Build and test
docker build -f Dockerfile.cpu -t whisperx:test .
docker run -d -p 8000:8000 whisperx:test

# Try it out
open http://localhost:8000
```

### Submitting a PR

When you're ready to contribute back:

1. Test everything works
2. Make sure docs are updated
3. Open PR from this branch to main
4. CI will test automatically

### Publishing to Docker Hub

See `DOCKER_DEPLOYMENT.md` for full setup instructions.

TL;DR:
- Add Docker Hub secrets to GitHub
- Push code
- Wait for build
- Images appear on Docker Hub

---

*Happy self-hosting! 🏠*
