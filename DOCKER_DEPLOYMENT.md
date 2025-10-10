# 🐳 WhisperX Docker Deployment Guide

*For the homelab enthusiast who wants clean, automated deployments*

## What's Been Added

This branch adds a complete Docker deployment stack for WhisperX with a modern web UI. Perfect for running in your homelab, NAS, or any server you've got laying around.

### Key Features

- **🌐 Web Interface**: Upload files via drag-and-drop, watch progress in real-time
- **📥 On-Demand Models**: Download only what you need (no 20GB pre-installed bloat)
- **🤖 Model Management**: Download and manage models right from the web UI
- **🎯 Speaker Diarization**: Built-in support (just need a Hugging Face token)
- **📊 Progress Tracking**: Know exactly what's happening with your transcriptions
- **🔄 Automated CI/CD**: Push code, GitHub Actions builds and publishes to Docker Hub

### What Changed From Base Repo

**New Files:**
- `Dockerfile` - GPU-capable image (NVIDIA CUDA 12.1.1)
- `Dockerfile.cpu` - CPU-only image for regular folks
- `docker-compose.yml` - Easy multi-service setup
- `api.py` - FastAPI backend with web UI support
- `web/index.html` - Modern web interface
- `requirements-api.txt` - API dependencies
- `.github/workflows/docker-publish.yml` - Automated builds
- `.dockerignore` - Keeps builds fast

**Modified Files:**
- Updated Python dependencies for compatibility

**Image Sizes:**
- CPU: ~3.8GB (PyTorch CPU is lean)
- GPU: ~14.7GB (CUDA + GPU PyTorch is chunky, but normal)

---

## 🚀 Quick Start

### Option 1: Use Pre-Built Images (Easiest)

Once this is on Docker Hub, just pull and run:

```bash
# CPU version (works anywhere)
docker run -d -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  your-username/whisperx:cpu-latest

# GPU version (needs NVIDIA GPU)
docker run -d --gpus all -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  your-username/whisperx:gpu-latest
```

**Notes:**
- `--privileged` is required for CTranslate2 on Docker Desktop (Windows/Mac). On Linux, you may be able to use `--security-opt seccomp=unconfined` instead.
- `-v ~/whisperx-models:/app/models` persists models between container restarts. Models downloaded via the web UI will be saved here!

Open http://localhost:8000 and you're good to go!

### Option 2: Build Locally

```bash
# Clone the repo
git clone https://github.com/m-bain/whisperX.git
cd whisperX
git checkout docker-image-web-interface

# Build CPU version
docker build -f Dockerfile.cpu -t whisperx:cpu .

# Build GPU version (if you've got the hardware)
docker build -f Dockerfile -t whisperx:gpu .

# Run it (with persistent model storage)
docker run -d -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  whisperx:cpu
```

### Option 3: Docker Compose (My Favorite)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  whisperx:
    image: your-username/whisperx:cpu-latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models      # Persist downloaded models
      - ./input:/input            # Drop files here
      - ./output:/output          # Transcriptions go here
    environment:
      - HF_TOKEN=${HF_TOKEN}      # For speaker diarization
    restart: unless-stopped
```

Then just: `docker-compose up -d`

---

## 🎛️ Configuration

### Environment Variables

| Variable | Default | What It Does |
|----------|---------|--------------|
| `HF_TOKEN` | - | Hugging Face token (needed for speaker diarization) |
| `DEVICE` | auto-detected | `cuda` or `cpu` |
| `COMPUTE_TYPE` | `float16` (GPU) / `int8` (CPU) | Model precision |
| `BATCH_SIZE` | `16` | Higher = faster but more memory |

### Volumes You'll Want

| Container Path | What It's For |
|---------------|---------------|
| `/app/models` | Cache downloaded models here (saves re-downloading) |
| `/input` | Drop audio files here |
| `/output` | Get your transcriptions here |

### Example with All the Options

```bash
docker run -d \
  --name whisperx \
  --privileged \
  -p 8000:8000 \
  -v ~/whisperx-models:/app/models \
  -v ~/whisperx-input:/input \
  -v ~/whisperx-output:/output \
  -e HF_TOKEN=hf_yourtoken \
  -e BATCH_SIZE=8 \
  --restart unless-stopped \
  your-username/whisperx:cpu-latest
```

---

## 📖 Using the Web Interface

1. **Open**: http://localhost:8000
2. **Drag & Drop**: Your audio/video file (or click to browse)
3. **Configure**:
   - Pick a model size (start with `base` for testing)
   - Set language (or leave blank for auto-detect)
   - Enable speaker diarization if you want (needs HF token)
4. **Hit Transcribe**: Watch the progress bar do its thing
5. **Download**: Get your results in JSON, SRT, VTT, TXT, or TSV

### Downloading Models

In the Model Management section:
- Select a model from the dropdown
- Click Download
- Watch it download with a progress bar
- Once downloaded, it's cached for next time

Model sizes:
- **tiny** (39MB) - Fast, okay accuracy
- **base** (142MB) - Good balance, my go-to for testing
- **small** (466MB) - Better quality
- **medium** (1.42GB) - High quality
- **large-v3** (2.87GB) - Best quality, slowest

---

## 🔧 API Usage

Full docs at http://localhost:8000/docs (Swagger UI)

### Quick Examples

```bash
# Transcribe a file
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@podcast.mp3" \
  -F "model=base" \
  -F "language=en"

# Check progress
curl http://localhost:8000/transcribe/TASK_ID

# List downloaded models
curl http://localhost:8000/models/list

# Download a model
curl -X POST http://localhost:8000/models/download \
  -H "Content-Type: application/json" \
  -d '{"model_name": "medium"}'
```

---

## 🎮 GPU Setup

If you've got an NVIDIA GPU and want to use it:

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Test It

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If you see your GPU info, you're golden!

### Run WhisperX with GPU

```bash
docker run -d --gpus all -p 8000:8000 your-username/whisperx:gpu-latest
```

---

## 🔄 Setting Up CI/CD (Publishing to Docker Hub)

Want to automate building and publishing? Here's how:

### 1. Get Docker Hub Ready

1. Sign up at https://hub.docker.com (free account works fine)
2. Go to Account Settings → Security
3. Create New Access Token
   - Name it something like "GitHub Actions"
   - Give it Read, Write, Delete permissions
4. Copy the token (you won't see it again!)

### 2. Configure GitHub Secrets

1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Add two secrets:
   - `DOCKER_HUB_USERNAME`: your Docker Hub username
   - `DOCKER_HUB_TOKEN`: the token you just created

### 3. Push and Watch the Magic

```bash
git add .
git commit -m "Add Docker deployment"
git push
```

GitHub Actions will:
- Build both CPU and GPU images
- Test them
- Push to Docker Hub
- Tag them properly

This takes about 30-40 minutes (building ML images is slow, grab a coffee ☕)

### 4. Making Releases

Want versioned images?

```bash
git tag v1.0.0
git push origin v1.0.0
```

This creates:
- `your-username/whisperx:1.0.0`
- `your-username/whisperx:1.0`
- `your-username/whisperx:1`
- Plus a GitHub Release with notes

---

## 🔍 Troubleshooting

### CTranslate2 Error: "cannot enable executable stack"

If you see this error when downloading models:
```
libctranslate2-*.so.*: cannot enable executable stack as shared object requires: Invalid argument
```

**Solution:** This is automatically handled by the container's entrypoint script which uses `setarch -R` to enable READ_IMPLIES_EXEC personality. You must still run with `--privileged` mode:

```bash
docker run -d -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  whisperx:cpu
```

**Why?** CTranslate2's shared libraries require executable stack permissions. Docker Desktop on Windows/Mac blocks this for security. The container uses `setarch -R` combined with `--privileged` mode to safely enable this functionality.

### Container Won't Start

```bash
# Check the logs
docker logs container-name

# Common issue: port already in use
netstat -an | grep 8000
```

### GPU Not Working

```bash
# Make sure nvidia-smi works in Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# If not, check your driver
nvidia-smi
```

### Models Downloading Slow

- Mount `/app/models` as a volume so you don't re-download
- Pre-download via the web UI
- Use a smaller model for testing

### Out of Memory

- Lower the batch size: `-e BATCH_SIZE=4`
- Use a smaller model (tiny or base)
- For CPU: Use `int8` compute type

### Web UI Won't Load

- Check if container is running: `docker ps`
- Check logs: `docker logs whisperx`
- Try accessing from container: `docker exec whisperx curl localhost:8000`

---

## 🛡️ Security Notes

Running this in your homelab? Cool. Exposing it to the internet? Let's be smart:

**Do:**
- ✅ Use a reverse proxy (nginx, Traefik, Caddy)
- ✅ Enable HTTPS (Let's Encrypt makes this easy)
- ✅ Keep your HF_TOKEN secret
- ✅ Update regularly (`docker pull` the latest)

**Don't:**
- ❌ Expose port 8000 directly to the internet
- ❌ Use weak/default passwords if you add auth
- ❌ Commit secrets to git

### Quick nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name whisperx.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Then use Certbot for HTTPS.

---

## 📊 Performance Tips

### Model Choice

For a 1-minute audio file:

| Model | CPU Time | GPU Time | Quality |
|-------|----------|----------|---------|
| tiny | ~30s | ~5s | Okay for testing |
| base | ~1min | ~8s | Good for most stuff |
| small | ~3min | ~12s | Better accuracy |
| medium | ~8min | ~20s | High quality |
| large-v3 | ~15min | ~35s | Best quality |

### Hardware Recommendations

**CPU Version:**
- Minimum: 4 cores, 8GB RAM
- Recommended: 8 cores, 16GB RAM
- Works on: Any x86_64 or ARM64 machine

**GPU Version:**
- Minimum: NVIDIA GPU with 4GB VRAM
- Recommended: NVIDIA GPU with 8GB+ VRAM
- Works on: Any NVIDIA GPU with CUDA support

---

## 🎯 Use Cases

What I use this for:
- 📺 Transcribing YouTube videos for notes
- 🎤 Meeting recordings
- 🎙️ Podcast episode notes
- 🎬 Adding subtitles to home videos
- 📞 Voicemail transcription

---

## 📚 Additional Resources

- **WhisperX Repo**: https://github.com/m-bain/whisperX
- **Original Paper**: https://arxiv.org/abs/2303.00747
- **Faster Whisper**: https://github.com/guillaumekln/faster-whisper
- **PyAnnote Audio**: https://github.com/pyannote/pyannote-audio

---

## 🤝 Contributing Back

Found a bug? Want to add a feature? 

1. Fork the repo
2. Make your changes on this branch
3. Test it with Docker
4. Open a PR

The CI/CD will test your changes automatically.

---

## 📝 Notes on Image Sizes

**Why is the GPU image 14.7GB?**

Yeah, it's chonky. Here's why:
- CUDA runtime: ~2.2GB (needed for GPU support)
- PyTorch with CUDA: ~6-7GB (GPU kernels are huge)
- Other ML libraries: ~3-4GB
- System packages: ~1GB

This is actually normal for GPU deep learning images. Compare:
- PyTorch official: 10-15GB
- TensorFlow GPU: 8-12GB
- Our WhisperX: 14.7GB ✓

The CPU version is only 3.8GB because PyTorch CPU is way smaller.

**Can we make it smaller?**

Not really without breaking stuff. We already:
- ✅ Use runtime images (not devel)
- ✅ Removed all build tools
- ✅ Don't pre-install models
- ✅ Multi-stage builds where possible

If you really need smaller, use the CPU version or accept the download time.

---

## 🎊 That's It!

You should now have a fully functional WhisperX instance running in Docker with a slick web UI. 

Questions? Issues? Open a GitHub issue or discussion.

Happy transcribing! 🎉

---

*Built with ❤️ for the self-hosted community*
