---
title: Constrained Refactor Gauntlet
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Constrained Refactor Gauntlet

An OpenEnv RL environment where an agent refactors a legacy Python codebase while obeying 150 cascading engineering rules.

**Reward = 0.50 × CodeScore + 0.35 × ComplianceScore + 0.15 × GreenScore**

## Hackathon

Meta PyTorch OpenEnv Hackathon — Long-Horizon Planning & Instruction Following

## Architecture

- **Base Model**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Training**: GRPO (Group Relative Policy Optimization) via Unsloth
- **Adapter**: [shreeyanshi03/constrained-refactor-adapter](https://huggingface.co/shreeyanshi03/constrained-refactor-adapter)
- **Track A**: Code Quality (lint, complexity, tests, module size)
- **Track B**: Compliance (150 cascading engineering rules)
- **Track C**: Green-Code Optimizer (energy efficiency via graphlet analysis + CPU/memory measurement)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Project info |
| `/health` | GET | Health check |
| `/health/green` | GET | Track C green subsystem status |
| `/docs` | GET | Interactive Swagger UI |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action in the environment |
| `/infer` | POST | Run trained agent (requires GPU) |
| `/dashboard/co2/{episode_id}` | GET | CO2 savings dashboard for an episode |

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn server:app --host 0.0.0.0 --port 7860`

## Training

```bash
# On GPU machine (Lightning Studio / Colab)
python training/verify_pipeline.py   # Verify setup
python training/train_grpo.py        # Full training run
```
