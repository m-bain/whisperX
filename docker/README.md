# Docker Deployment Files

This directory contains all Docker-related files for WhisperX deployment.

## 📁 Structure

```
docker/
├── Dockerfile              # GPU-enabled image (NVIDIA CUDA)
├── Dockerfile.cpu          # CPU-only image (with CTranslate2 fix)
├── docker-compose.yml      # Multi-service orchestration
├── entrypoint.sh          # Fixes CTranslate2 executable stack issue
├── .dockerignore          # Build optimization
├── api.py                 # FastAPI backend
├── requirements-api.txt   # API dependencies
├── DOCKERHUB_SETUP.md     # Docker Hub automation guide
└── web/
    └── index.html         # Modern web interface
```

## 🚀 Quick Start

### Build from Root Directory

```bash
# From repository root
cd /path/to/whisperX

# Build CPU version
docker build -f docker/Dockerfile.cpu -t whisperx:cpu .

# Build GPU version
docker build -f docker/Dockerfile -t whisperx:gpu .
```

### Run Container

```bash
# CPU version (with persistent model storage)
docker run -d -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  whisperx:cpu

# GPU version
docker run -d --gpus all -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  whisperx:gpu
```

**Notes:** 
- `--privileged` is required for CTranslate2 on Docker Desktop (Windows/Mac)
- `-v ~/whisperx-models:/app/models` persists downloaded models between restarts

Access the web interface at http://localhost:8000

### Using Docker Compose

```bash
# From repository root
cd /path/to/whisperX

# Start CPU API service
docker-compose -f docker/docker-compose.yml --profile api-cpu up -d

# Start GPU API service
docker-compose -f docker/docker-compose.yml --profile api-gpu up -d
```

## 📖 Full Documentation

See the complete deployment guide: [DOCKER_DEPLOYMENT.md](../DOCKER_DEPLOYMENT.md)

## 🔧 Configuration

### Environment Variables

- `HF_TOKEN` - Hugging Face token for speaker diarization
- `DEVICE` - `cuda` or `cpu` (auto-detected)
- `COMPUTE_TYPE` - Model precision (`float16`, `int8`)
- `BATCH_SIZE` - Processing batch size (default: 16)

### Volumes

- `/app/models` - Model cache (recommended for persistence)
- `/input` - Input audio files
- `/output` - Transcription outputs

## 🎯 Available Services

### docker-compose.yml Profiles

| Profile | Service | Port | Description |
|---------|---------|------|-------------|
| `api-cpu` | whisperx-api-cpu | 8005 | Web API (CPU) |
| `api-gpu` | whisperx-api-gpu | 8005 | Web API (GPU) |
| `cpu` | whisperx-cpu | - | CLI only (CPU) |
| `gpu` | whisperx-gpu | - | CLI only (GPU) |

## 📦 Image Sizes

- **CPU**: ~3.8GB (PyTorch CPU + dependencies)
- **GPU**: ~14.7GB (CUDA 12.1.1 + PyTorch GPU + dependencies)

## 🔄 CI/CD

GitHub Actions workflow automatically builds and publishes images to Docker Hub on push/tag.

See: `.github/workflows/docker-publish.yml`

## 🛠️ Development

### Testing Local Changes

```bash
# Build with local changes
docker build -f docker/Dockerfile.cpu -t whisperx:test .

# Run and test
docker run -it -p 8000:8000 whisperx:test

# Check logs
docker logs <container-id>
```

### Debugging

```bash
# Enter running container
docker exec -it <container-id> /bin/bash

# Check installed packages
docker exec <container-id> pip list

# View API logs
docker logs -f whisperx-api-cpu
```

## 📝 Notes

- **Build Context**: Always build from repository root (parent directory)
- **Paths**: All COPY commands reference parent directory structure
- **Organization**: Keeps Docker files separate from main codebase
- **PR Friendly**: Clean separation for upstream contributions

## 🆘 Troubleshooting

### Build Fails

```bash
# Clear Docker cache
docker builder prune -a

# Check context
docker build -f docker/Dockerfile.cpu -t whisperx:test . --no-cache
```

### Container Won't Start

```bash
# Check logs
docker logs whisperx-api-cpu

# Verify port not in use
netstat -an | grep 8000
```

## 📚 Related Documentation

- [DOCKER_DEPLOYMENT.md](../DOCKER_DEPLOYMENT.md) - Complete deployment guide
- [CHANGELOG.md](../CHANGELOG.md) - What's new in this branch
- [README.md](../README.md) - Main WhisperX documentation
