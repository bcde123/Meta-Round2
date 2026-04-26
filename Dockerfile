# ── Stage 1: Environment Server ──────────────────────────────────────────────
# Lightweight image for the OpenEnv-compliant environment + inference endpoint.
# The trained LoRA adapter is downloaded from HuggingFace Hub at startup.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

# System deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (two layers for caching) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Fix for KeyError: 'getpwuid(): uid not found: 1000'
# Hugging Face Spaces runs as UID 1000 without a corresponding /etc/passwd entry.
# Create a dummy passwd entry so getpass.getuser() works.
RUN echo "huggingface:x:1000:1000:HuggingFace user:/home/huggingface:/bin/sh" >> /etc/passwd
# Torch Inductor tries to get the username for caching.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
ENV TORCHINDUCTOR_DISABLE=1
ENV USER=huggingface
ENV LOGNAME=huggingface

EXPOSE 7860

# Health check for container orchestrators / HF Spaces
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD uvicorn server:app --host 0.0.0.0 --port 7860 & python training/train_grpo.py
