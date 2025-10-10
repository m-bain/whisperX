# Docker Hub Automation Setup Guide

This guide explains how to set up automated Docker image builds and publishing to Docker Hub using GitHub Actions.

## Prerequisites

1. **Docker Hub Account**: Create an account at [hub.docker.com](https://hub.docker.com)
2. **Docker Hub Access Token**: Generate from Account Settings → Security → Access Tokens
3. **GitHub Repository**: Your WhisperX fork with the docker branch

## Step 1: Create Docker Hub Repositories

Go to Docker Hub and create two repositories:
- `whisperx-cpu` - For CPU-only images
- `whisperx-gpu` - For GPU-enabled images

Set both to **Public** for easy access.

## Step 2: Configure GitHub Secrets

In your GitHub repository, go to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username | Used for login and image naming |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token | Authentication for pushing images |

## Step 3: Workflow Configuration

The workflow file is already set up at `.github/workflows/docker-publish.yml`. It will:

### Trigger On:
- **Push to main branch** - Builds and pushes with `latest` tag
- **Push to docker-image-web-interface** - Builds and pushes with branch name tag
- **Tags matching `v*.*.*`** - Creates versioned releases (e.g., `v1.0.0`)
- **Pull requests** - Builds only (no push)
- **Manual trigger** - Via GitHub Actions UI

### Build Matrix:
- **CPU Image**: Multi-arch (linux/amd64, linux/arm64)
- **GPU Image**: Single arch (linux/amd64 only, due to CUDA)

### Tags Generated:
```bash
# For CPU image:
yourusername/whisperx:cpu
yourusername/whisperx:cpu-latest
yourusername/whisperx:cpu-v1.0.0  # on version tags

# For GPU image:
yourusername/whisperx:gpu
yourusername/whisperx:gpu-latest
yourusername/whisperx:latest  # Main tag points to GPU
yourusername/whisperx:gpu-v1.0.0  # on version tags
```

## Step 4: Test the Workflow

### Option A: Push to Branch
```bash
git add .
git commit -m "feat: Add Docker Hub automation"
git push origin docker-image-web-interface
```

Go to GitHub Actions tab to watch the build progress.

### Option B: Manual Trigger
1. Go to **Actions** tab in GitHub
2. Select "Build and Push Docker Images"
3. Click **Run workflow**
4. Select branch and click **Run workflow**

## Step 5: Verify Images

After the workflow completes successfully:

1. Check Docker Hub repositories for new images
2. Pull and test locally:
```bash
# CPU image
docker pull yourusername/whisperx:cpu-latest
docker run -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  yourusername/whisperx:cpu-latest

# GPU image
docker pull yourusername/whisperx:gpu-latest
docker run --gpus all -p 8000:8000 --privileged \
  -v ~/whisperx-models:/app/models \
  yourusername/whisperx:gpu-latest
```

## Step 6: Create a Release (Optional)

To create a versioned release with automatic GitHub Release notes:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This will:
1. Build and push versioned images (e.g., `:cpu-v1.0.0`, `:gpu-v1.0.0`)
2. Run automated tests
3. Create a GitHub Release with changelog

## Workflow Features

### ✅ Multi-Architecture Support
- **CPU images**: Built for both amd64 and arm64 (Apple Silicon support!)
- **GPU images**: amd64 only (NVIDIA CUDA requirement)

### ✅ Build Caching
- Uses GitHub Actions cache for faster rebuilds
- Only rebuilds changed layers

### ✅ Automated Testing
- Tests image import after build
- Verifies WhisperX can be imported successfully

### ✅ GitHub Releases
- Auto-generates release notes for version tags
- Includes usage instructions and changelog

## Troubleshooting

### Build Fails with Authentication Error
- Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set correctly
- Regenerate Docker Hub access token if needed

### Images Not Appearing on Docker Hub
- Check if repositories exist: `whisperx-cpu` and `whisperx-gpu`
- Verify workflow completed successfully in Actions tab
- Check for errors in workflow logs

### Multi-Arch Build Fails
- ARM64 builds may take longer (emulation)
- Check if QEMU is set up correctly (handled automatically by workflow)

### GPU Build Takes Too Long
- Normal - GPU images are ~14GB due to CUDA runtime
- Consider limiting to amd64 only (already configured)

## Customization

### Change Image Names
Edit `.github/workflows/docker-publish.yml`:
```yaml
env:
  IMAGE_NAME: your-custom-name  # Change this
```

### Add More Architectures
Modify the platforms in the workflow:
```yaml
platforms: linux/amd64,linux/arm64,linux/arm/v7
```

### Change Trigger Branches
Modify the `on.push.branches` section:
```yaml
on:
  push:
    branches:
      - main
      - develop  # Add your branches here
```

## Next Steps

1. ✅ Set up Docker Hub account and tokens
2. ✅ Configure GitHub secrets
3. ✅ Push to trigger first build
4. ✅ Verify images on Docker Hub
5. 📝 Update README with your Docker Hub username
6. 🎉 Share with the community!

## Resources

- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [Multi-Platform Images Guide](https://docs.docker.com/build/building/multi-platform/)
