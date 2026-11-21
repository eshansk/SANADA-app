# Multi-stage production Dockerfile for SANADA Machine Listening API
# Compatible with campus clusters, Docker, and non-root Kubernetes/Docker hosts

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HF_HOME=/tmp/hf_cache
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Install system dependencies for audio decoding and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure writable directories exist with correct permissions for user 1000
RUN mkdir -p /app/backend/static/stems /app/models/checkpoints /tmp/hf_cache \
    && chown -R user:user /app /tmp/hf_cache

# Copy source modules, scripts, backend, models, inference module, and frontend
COPY --chown=user:user sanada_inference.py .
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user scripts/ ./scripts/
COPY --chown=user:user frontend/ ./frontend/

# Switch to non-root user
USER user

EXPOSE 8000

# Health check to ensure service vitality
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Launch startup verification and ASGI production server
CMD ["sh", "-c", "python scripts/download_checkpoints.py && uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
