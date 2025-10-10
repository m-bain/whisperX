# Lean WhisperX Dockerfile - GPU/CPU with on-demand model downloads
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Fix for CTranslate2 executable stack issue
ENV SELINUX=permissive
# Set Hugging Face cache to use mounted volume
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY docker/requirements-api.txt .
COPY pyproject.toml .
COPY MANIFEST.in .

# Install Python dependencies (this layer will be cached)
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements-api.txt

# Install WhisperX dependencies without heavy models
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    "transformers>=4.48.0" \
    "faster-whisper>=1.1.1" \
    "pyannote-audio>=3.3.2,<4.0.0" \
    "nltk>=3.9.1" \
    "numpy>=2.0.2" \
    "pandas>=2.2.3" \
    "ctranslate2<4.5.0" \
    "onnxruntime>=1.19"

# Copy application code
COPY whisperx/ ./whisperx/
COPY docker/api.py .
COPY docker/web/ ./web/

# Install WhisperX in development mode (no dependencies, already installed)
RUN pip3 install --no-deps -e .

# Create directories with proper permissions
RUN mkdir -p /input /output /app/models && \
    chmod 755 /app /input /output /app/models

# Copy and setup entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script that sets READ_IMPLIES_EXEC personality
ENTRYPOINT ["/entrypoint.sh"]