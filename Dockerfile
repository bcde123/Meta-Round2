# ── Stage 1: Environment Server ──────────────────────────────────────────────
# Lightweight image for the OpenEnv-compliant environment + inference endpoint.
# The trained LoRA adapter is downloaded from HuggingFace Hub at startup.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (two layers for caching) ─────────────────────
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY environment/ ./environment/
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY training/ ./training/

# ── Environment variables ────────────────────────────────────────────────────
# HF_TOKEN must be set at runtime for adapter download
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 7860

# Health check for container orchestrators / HF Spaces
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
