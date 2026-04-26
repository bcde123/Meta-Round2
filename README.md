---
title: Constrained Refactor Gauntlet
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Constrained Refactor Gauntlet

An **OpenEnv**-compatible RL environment where an agent refactors a legacy Python codebase while obeying **150 cascading engineering rules**.

$$R_{total} = (W_{test} \cdot S_{test}) \times \left( \frac{1}{N} \sum_{i=1}^{N} C_i \right) - P_{efficiency} - P_{hack}$$

## 🔗 Submission Links

| Resource | Link |
|----------|------|
| 🤗 **HF Space (live env)** | [s123hree/constrained-refactor-gauntlet-a100](https://huggingface.co/spaces/s123hree/constrained-refactor-gauntlet-a100) |
| 🤗 **Trained Adapter** | [shreeyanshi03/constrained-refactor-adapter-1.5b](https://huggingface.co/shreeyanshi03/constrained-refactor-adapter-1.5b) |
| 📓 **Colab Training Notebook** | [`notebooks/train_grpo.ipynb`](notebooks/train_grpo.ipynb) |
| 📝 **Blog Post (writeup)** | _TODO: paste HF blog URL here_ |
| 🎥 **2-min Video Demo** | _TODO: paste YouTube URL here_ |
| 📊 **Training Plots** | [`assets/training_curves.png`](assets/training_curves.png) |

## 🎯 Hackathon

**OpenEnv India Hackathon 2026** — Meta PyTorch — Long-Horizon Planning & Instruction Following

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Env[Environment Server]
        Reset["/reset"] --> EpisodeGen[Episode Generator]
        EpisodeGen --> Corrupt[Corruption Pipeline]
        Corrupt --> State[Initial State]
        State --> Step[/step]
        Step --> Eval[Evaluation Engine]
        Eval --> Reward[Reward Function]
        Reward --> Step
    end
    subgraph Train[Training Pipeline]
        Model[(Base Model\nQwen/Qwen2.5‑Coder‑7B‑Instruct)] --> LoRA[LoRA Adapters]
        LoRA --> GRPO[GRPO Trainer]
        GRPO --> Dataset[Generated Episodes]
        Dataset --> GRPO
    end
    subgraph Eval[Evaluation Tracks]
        TrackA[Track A – Code Quality]
        TrackB[Track B – Compliance]
        TrackC[Track C – Green‑Code]
        Reward --> TrackA & TrackB & TrackC
    end
    Env --> Train
    Train --> Infer[/infer]
```

### Core Components

- **Episode Generator (`environment/episode_generator.py`)** – Loads a clean codebase, applies a random subset of corruptions (circular imports, cryptic renames, dead code, hard‑coded secrets, etc.), and produces the initial episode state together with an active set of engineering rules. Difficulty is scaled via a `CurriculumManager` based on recent agent performance.
- **Curriculum Manager** – Tracks rolling reward history (last 150 episodes) and escalates rule count (up to 150) once the agent consistently exceeds a 0.7 success threshold.
- **FastAPI Server (`server.py`)** – Exposes a standard RL interface:
  - `GET /` – Project info
  - `GET /health` – Health check
  - `POST /reset` – Start a new episode
  - `POST /step` – Submit an action (XML‑formatted file edits)
  - `POST /infer` – Run the trained agent on the current state (GPU required)
  - `GET /dashboard/co2/{episode_id}` – Visualise CO₂‑savings from Track C
- **Evaluation Engine** – Implements three orthogonal tracks that feed the final reward:
  - **Track A – Code Quality** – Fast AST‑based lint, cyclomatic‑complexity, module‑size, doc‑string and type‑hint coverage.
  - **Track B – Compliance** – Checks against the 150 engineering standards defined in `ENGINEERING_STANDARDS.md`.
  - **Track C – Green‑Code** – Graphlet‑analysis + CPU/memory profiling to estimate energy‑efficiency and translate it into a CO₂‑saving score.
- **Training Pipeline (`training/train_grpo.py`)** – Uses **Unsloth** to load the base model with 4‑bit Quant‑LLM (QLoRA) and wraps it with LoRA adapters. Episodes are generated on‑the‑fly, the model produces several completions per prompt, and the custom `reward_function` scores each completion using the multiplicative formula (plus a formatting bonus). GRPO then performs a relative‑policy update.
- **Inference (`inference.py`)** – Loads the final LoRA adapter, receives the current episode state via `/infer`, and returns the best edit payload.

### Reward Components

| Component | Definition | Verification |
|-----------|------------|-------------|
| **Test Score (S_test)** | Does the refactored code still function correctly? | Binary gate: `1.0` if all files parse & tests pass, `0.0` for any failure. |
| **Compliance Score (C_i)** | Did the model follow the active engineering rules? | Per‑rule AST parsing to verify exact structural constraints (70% rule‑engine + 30% direct AST). |
| **Efficiency Penalty (P_efficiency)** | Did the model take too many steps? | Subtracts `0.01` per step/edit to encourage direct, minimal fixes. |
| **Hack Penalty (P_hack)** | Did the model try to cheat the environment? | Immediate `−1.0` reward and episode termination. |

### Anti‑Cheating Layers

1. **Binary Execution Gate** – If ANY file in the codebase has a `SyntaxError`, the test multiplier drops to **zero**. The agent gets no points for “clean” code that doesn’t compile.
2. **Protected File Lockdown** – Test infrastructure files (`conftest.py`, `test_*.py`, `pytest.ini`, `setup.cfg`) cannot be edited. Any attempt triggers `P_hack = −1.0`.
3. **Test Stub Detection** – Creating functions like `def test_all(): return True` is flagged as a hack via AST inspection.
4. **Forbidden Names** – Specific naming conventions (e.g., `varelunixo`, `xhackbypass`) trigger immediate penalties.
5. **Assertion Guard** – Deleting all `assert` statements from a file that originally contained them is treated as cheating.

## 📚 API Endpoints

| Endpoint                               | Method | Description                               |
|----------------------------------------|--------|-------------------------------------------|
| `/`                                    | GET    | Project information                       |
| `/health`                              | GET    | Simple health check                       |
| `/health/green`                        | GET    | Status of the Green‑Code subsystem        |
| `/docs`                                | GET    | Swagger UI for the FastAPI server          |
| `/reset`                               | POST   | Initialise a new episode                  |
| `/step`                               | POST   | Submit an action (file edit)               |
| `/infer`                               | POST   | Run the trained agent (GPU required)      |
| `/dashboard/co2/{episode_id}`          | GET    | CO₂‑savings dashboard for an episode      |

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bcde123/Meta-Round2.git
   cd Meta-Round2
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the environment server**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 7860
   ```
4. **(Optional) Train the model** – see the Training section below.

## 🚀 Training

```bash
# Verify the environment (CPU‑only quick check)
python training/verify_pipeline.py

# Full GRPO training (GPU, 200 episodes)
python training/train_grpo.py
```
The script:
1. Generates a synthetic dataset of corrupted episodes.
2. Loads the Qwen‑2.5‑Coder base model via Unsloth.
3. Attaches LoRA adapters (`r=32`).
4. Runs GRPO with a custom reward that combines Track A, B, C and a format‑bonus.
5. Saves the final adapter to `grpo_output/final_adapter/`.

## 📈 Inference & Evaluation

```bash
python inference.py   # loads the saved adapter and starts a demo loop
```
* The agent receives the current episode via the server, predicts the next edit, and the server applies it.
* After the episode finishes, the three tracks emit a detailed score breakdown and, for Track C, a CO₂‑savings estimate displayed at `/dashboard/co2/<episode_id>`.

## 🤝 Contributing

- Follow the **PEP‑8** style guide and keep docstrings.
- Add new corruptions to `EpisodeGenerator` as separate methods.
- Extend `ENGINEERING_STANDARDS.md` with additional rule definitions – the compliance checker will pick them up automatically.
- Open a PR with a clear description and update the changelog.

## 📊 Results

We trained **Qwen2.5-Coder-1.5B-Instruct** with QLoRA (`r=16`) using GRPO on a single A100-80GB.

| Metric | Value |
|--------|-------|
| Base model | Qwen2.5-Coder-1.5B-Instruct |
| LoRA rank / alpha | 16 / 16 |
| Training steps | 200 |
| Generations per step | 4 |
| Hardware | NVIDIA A100-SXM4-80GB |
| Wall-clock training time | ~25 min |

**Training curves** (loss ↓, reward ↑):

![Training curves](assets/training_curves.png)

| | Before training | After training |
|--|---------------|---------------|
| Mean episode reward | _baseline_ | _final_ |
| Test-pass rate | _baseline_ | _final_ |
| Compliance score | _baseline_ | _final_ |
| Avg. steps to solve | _baseline_ | _final_ |

> Numbers will be filled in once the training run completes. Plots in `assets/` are saved automatically by `training/train_grpo.py` from `trainer.state.log_history`.

Reproduce in Colab: [`notebooks/train_grpo.ipynb`](notebooks/train_grpo.ipynb).

## 🧩 OpenEnv Compatibility

This environment follows the [OpenEnv](https://github.com/meta-pytorch/openenv) spec:

- **Manifest**: [`openenv.yaml`](openenv.yaml)
- **Standard endpoints**: `POST /reset`, `POST /step`, `GET /health`
- **Observation space**: `{ files: dict, violation_report: dict, steps_remaining: int, curriculum_level: int }`
- **Action space**: tools `[read_file, edit_file, run_tests, check_compliance]`
- **Reward range**: `[0.0, 1.0]`
- **Max episode length**: 70 steps

## 📜 License

This project is released under the **Apache‑2.0 License**. Feel free to fork, modify, and submit improvements.

---
*Created with ❤️ by the Meta‑Round 2 team.*
