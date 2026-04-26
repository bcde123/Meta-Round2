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

# ── CUDA library discovery for bitsandbytes ──────────────────────────────────
# bitsandbytes loads libbitsandbytes_cuda<version>.so via ctypes.CDLL, which in
# turn depends on libcudart.so.12 from CUDA 12.4. On HF Spaces the container
# runs as UID 1000, so unsloth's runtime fallback `ldconfig /usr/lib64-nvidia`
# fails with "Permission denied" → bnb fails to import → unsloth then dies on
# `name 'bnb' is not defined`. Populate the loader cache at build time (where
# we ARE root) and set LD_LIBRARY_PATH explicitly so bnb can dlopen its libs
# without needing root at runtime.
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda/lib64:/usr/lib64-nvidia
# Force bitsandbytes to pick the CUDA 12.4 library that ships in the wheel,
# matching the pytorch:2.6.0-cuda12.4 base image. Without this it tries to
# auto-detect from `nvidia-smi`, which is flaky in non-interactive containers.
ENV BNB_CUDA_VERSION=124
RUN ldconfig

# Smoke-test bitsandbytes during build. HF Spaces builders are CPU-only, so
# `torch.cuda.is_available()` is False and bnb's auto-loader picks the CPU
# library — that won't tell us whether the CUDA shim works on the GPU node.
# Instead, dlopen libbitsandbytes_cuda124.so directly via ctypes: that does
# NOT require a live GPU (no kernel launch happens), only that libcudart.so.12
# and friends are reachable on LD_LIBRARY_PATH. If they're not, the build
# fails here with the real error instead of producing the cryptic runtime
# `name 'bnb' is not defined` we hit before.
RUN python -c "import bitsandbytes, ctypes, pathlib; \
pkg = pathlib.Path(bitsandbytes.__file__).parent; \
so = pkg / 'libbitsandbytes_cuda124.so'; \
print('dlopen', so); \
ctypes.CDLL(str(so)); \
print('bnb', bitsandbytes.__version__, 'CUDA shim loads OK')"

# ── Patch unsloth's bnb fallback path ────────────────────────────────────────
# unsloth/__init__.py has a footgun: if `import bitsandbytes as bnb` raises,
# the bare `except:` swallows it but `bnb` is never bound, and the *very next*
# except block calls `importlib.reload(bnb)` — which raises `NameError: name
# 'bnb' is not defined` and propagates out of `import unsloth` entirely.
# Wrap that single reload call so unsloth can finish importing in 16-bit mode
# even when bnb has any runtime issue, instead of taking the whole trainer
# (and our diagnostic logging) down with a cryptic NameError.
RUN python -c "import pathlib; \
p = pathlib.Path('/opt/conda/lib/python3.11/site-packages/unsloth/__init__.py'); \
src = p.read_text(); \
old = '        importlib.reload(bnb)\n        importlib.reload(triton)'; \
new = '        try: importlib.reload(bnb)\n        except NameError: pass  # bnb undefined if its import failed earlier\n        importlib.reload(triton)'; \
assert old in src, 'unsloth bnb-reload pattern not found — wheel changed shape'; \
p.write_text(src.replace(old, new, 1)); \
print('Patched unsloth __init__.py: guarded importlib.reload(bnb)')"

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
# Training checkpoints + plots: /app is not writable for UID 1000 on HF Spaces.
ENV GRPO_OUTPUT_DIR=/tmp/grpo_output
# unsloth_zoo defaults to cwd-relative "unsloth_compiled_cache" → PermissionError
# when cwd is /app. Point compile artifacts at /tmp instead.
ENV UNSLOTH_COMPILE_LOCATION=/tmp/unsloth_compiled_cache
# Note: GREEN_PROFILE_MODE is intentionally NOT set here. Training picks
# "compile" via os.environ.setdefault in train_grpo.py; the live API server
# keeps the default "runtime" so /step and /dashboard/co2 do real profiling.

# Fix for KeyError: 'getpwuid(): uid not found: 1000'
# Hugging Face Spaces runs as UID 1000 without a corresponding /etc/passwd entry.
# Create a dummy passwd entry AND a writable home dir. Without the home dir,
# triton's autotune cache (/home/huggingface/.triton/cache) blows up at
# `os.makedirs(...)` time during `import bitsandbytes` → which trips
# `from .nn.triton_based_modules import ...` → which trips
# `@triton.autotune(...)` → PermissionError. That cascades into
# `import unsloth` failing through transformers.integrations.bitsandbytes.
RUN echo "huggingface:x:1000:1000:HuggingFace user:/home/huggingface:/bin/sh" >> /etc/passwd \
    && mkdir -p /home/huggingface/.triton/cache /home/huggingface/.cache \
    && chmod -R 1777 /home/huggingface
# Make every cache-bearing tool point at a writable place explicitly. We can't
# rely on `~` resolution alone because `pwd.getpwuid()` and `expanduser('~')`
# both still return `/home/huggingface` on HF Spaces; setting HOME wins over
# both, and individual *_CACHE_DIR vars override even that.
ENV HOME=/home/huggingface
ENV XDG_CACHE_HOME=/tmp/xdg_cache
ENV TRITON_CACHE_DIR=/tmp/triton_cache
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
    'mkdir -p /tmp/torch_inductor /tmp/hf_home /tmp/triton_cache /tmp/xdg_cache /tmp/grpo_output /tmp/unsloth_compiled_cache' \
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
