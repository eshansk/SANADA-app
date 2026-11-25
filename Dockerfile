FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HF_HOME=/tmp/hf_cache
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps clearvoice

RUN mkdir -p /app/backend/static/stems /app/models/checkpoints /tmp/hf_cache \
    && chown -R user:user /app /tmp/hf_cache

COPY --chown=user:user sanada_inference.py .
COPY --chown=user:user dev_server.py .
COPY --chown=user:user start_system.sh .
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user scripts/ ./scripts/
COPY --chown=user:user frontend/ ./frontend/

USER user

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

CMD ["sh", "-c", "python scripts/download_checkpoints.py && uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
