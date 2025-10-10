#!/bin/bash

# WhisperX Docker Image Builder
# This script builds and tags Docker images for distribution

set -e

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-yourusername}"
IMAGE_NAME="whisperx-enhanced"
VERSION="${VERSION:-latest}"

echo "🐳 Building WhisperX Enhanced Docker Images"
echo "============================================"

# Build GPU version
echo "📦 Building GPU version..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} .
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu .

# Build CPU version
echo "📦 Building CPU version..."
docker build -f Dockerfile.cpu -t ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu .

# Tag with version
if [ "$VERSION" != "latest" ]; then
    docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:latest ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
fi

echo "✅ Build completed!"
echo ""
echo "Built images:"
docker images | grep ${DOCKER_USERNAME}/${IMAGE_NAME}

echo ""
echo "🚀 To push to Docker Hub:"
echo "docker login"
echo "docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:gpu"
echo "docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:cpu"

echo ""
echo "📋 To run locally:"
echo "docker run -p 8005:8000 --gpus all ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"