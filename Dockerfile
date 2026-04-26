# ── Green-Code Optimizer — Environment Image ────────────────────────────────
# OpenEnv-compatible RL env that trains a code agent to refactor Python for
# energy efficiency (CPU + memory). Includes graphlet analyzer, runtime
# profiler, and CO2-savings dashboard. The trained LoRA adapter is downloaded
# from HuggingFace Hub at startup.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

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
# Avoid CUDA fragmentation OOM when bnb 4-bit loads the base model right
# next to a peft + LoRA training graph. Set in env (not just Python) so it
# applies to every subprocess train_grpo.py spawns.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep HF caches on /tmp so the read-only HF Spaces home doesn't bite us.
ENV HF_HOME=/tmp/hf_home
ENV TRANSFORMERS_CACHE=/tmp/hf_home
ENV WANDB_DISABLED=true
# Note: GREEN_PROFILE_MODE is intentionally NOT set here. Training picks
# "compile" via os.environ.setdefault in train_grpo.py; the live API server
# keeps the default "runtime" so /step and /dashboard/co2 do real profiling.

# Fix for KeyError: 'getpwuid(): uid not found: 1000'
# Hugging Face Spaces runs as UID 1000 without a corresponding /etc/passwd entry.
# Create a dummy passwd entry so getpass.getuser() works.
RUN echo "huggingface:x:1000:1000:HuggingFace user:/home/huggingface:/bin/sh" >> /etc/passwd
# Torch Inductor tries to get the username for caching.
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
ENV TORCHINDUCTOR_DISABLE=1
ENV USER=huggingface
ENV LOGNAME=huggingface

# ── Entrypoint that survives a training crash ────────────────────────────────
# - Starts uvicorn in the foreground so SIGTERM reaches it cleanly.
# - Tees training logs to /tmp/train.log AND stdout, prefixed for clarity.
# - If training crashes (OOM, import error, …) we log it but keep the API up
#   so judges can still hit /demo, /reset, /step, /docs, etc.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -u' \
    'mkdir -p /tmp/torch_inductor /tmp/hf_home' \
    'echo "[entrypoint] starting GRPO training in background"' \
    '(' \
    '  python -u training/train_grpo.py 2>&1 | sed -u "s/^/[train] /"' \
    '  ec=${PIPESTATUS[0]}' \
    '  if [ "$ec" != "0" ]; then echo "[entrypoint] training exited with code $ec — API will keep running"; fi' \
    ') &' \
    'echo "[entrypoint] starting uvicorn on :7860"' \
    'exec uvicorn server:app --host 0.0.0.0 --port 7860 --log-level info' \
    > /usr/local/bin/start.sh \
    && chmod +x /usr/local/bin/start.sh

EXPOSE 7860

# Health check for container orchestrators / HF Spaces
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["/usr/local/bin/start.sh"]
