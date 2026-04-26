---
title: Green-Code Optimizer
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
---

# 🌱 Green-Code Optimizer

> **An RL agent that refactors Python code for energy efficiency, not readability — and tells you exactly how much CO₂ it saves.**

[![HF Space](https://img.shields.io/badge/🤗_Space-Live-blue)](https://huggingface.co/spaces/s123hree/green-code-optimizer-a100)
[![Repository](https://img.shields.io/badge/Repository-GitHub-black)](https://github.com/bcde123/Meta-Round2)
[![Blog Post](https://img.shields.io/badge/Blog_Post-GitHub-green)](https://github.com/bcde123/Meta-Round2/blob/main/blog_post.md)

---

## 🎯 The Problem

AI workloads are projected to consume **2–4 % of global electricity by 2030**. Most of that goes to *training*, but **inference at scale** — the same algorithms running millions of times a day in production — is the silent giant. A single inefficient nested loop in a hot path, replicated across millions of executions, can add up to *real* CO₂.

Meanwhile, almost every existing code-refactoring tool optimises for **readability** (line length, naming, type hints). **None of them refactor for energy.**

> **What if we trained an RL agent whose only goal was to make Python *cheaper to run* — measured in CPU cycles, memory footprint, and ultimately grams of CO₂?**

That's the Green-Code Optimizer.

---

## 💡 The Pitch

| Existing refactoring tools | Green-Code Optimizer |
|----------------------------|----------------------|
| Optimise for readability   | **Optimise for energy** |
| Style / naming / lint      | **CPU time + peak memory** |
| Subjective rules            | **Measurable, runtime-grounded reward** |
| Outputs cleaner code        | **Outputs cheaper code + a CO₂ dashboard** |

The agent receives a **negative reward for high peak memory and high execution time**, and a positive reward for the *opposite*. It uses **graphlet analysis** to represent the program's control-flow structure, so it learns which structural patterns (nested loops, function calls inside loops, deep branching) are expensive — and swaps them for cheap alternatives (vectorised ops, comprehensions, hoisted invariants).

---

## 👀 Concrete Example

A real corruption from the env (Level-1 episode):

**Before** — a list comprehension is exploded into an append-loop, *and* a 50-iteration energy-waste loop is injected at the top:
```python
def get_priority_users(users):
    _energy_waste = []
    for _ in range(50):
        _energy_waste.append(list(range(100)))
    out = []
    for u in users:
        out.append(u.upper())
    return out
```

**After** — the agent's target refactor:
```python
def get_priority_users(users):
    return [u.upper() for u in users]
```

**Impact (at 10 000 runs/day):** ~**1.1 kg CO₂/year** saved per call site. With one trained agent and a few thousand call sites, you're talking about *trees worth* of carbon.

---

## ⚙️ How It Works

```mermaid
flowchart LR
    A["Corrupted /<br/>energy-inefficient<br/>Python codebase"] --> B[Episode Generator]
    B --> C["RL Agent<br/>(Qwen-1.5B + LoRA)"]
    C -->|edits| D[Updated codebase]
    D --> E1[Graphlet Analyzer]
    D --> E2[CPU + Memory Profiler]
    D --> E3[Compliance / Test Gate]
    E1 --> R[GRPO Reward]
    E2 --> R
    E3 --> R
    R -->|policy update| C
    D --> F["CO₂ Dashboard<br/>kg/year · trees · car-km"]
```

### 1. Episode Generation with **Curriculum-Driven Corruption**

Each episode loads a real Python codebase and applies **energy-degrading corruptions** that scale with the agent's skill:

| Curriculum level | Energy corruptions | Readability corruptions | Active rules |
|-----------------:|:-------------------|:------------------------|-------------:|
| 1 (warmup) | 1 | 1 | 20 |
| 2 | 2 | 2 | 60 |
| 3 | 3 (all) | 3 | 100 |
| 4 (expert) | 3 (all, 2× passes) | 4 | 140 |

The `CurriculumManager` escalates the level once 3 consecutive 50-episode windows all average a reward > 0.7 — so the agent's *own progress* drives the difficulty.

### 2. Graphlet Analysis — `environment/graphlet_analyzer.py`
Parses each file into an AST and detects 4 classes of expensive control-flow graphlets:

| Graphlet | Cost weight | Example |
|----------|------------:|---------|
| `NestedLoop` | 3.0 | `for i: for j: ...` |
| `LoopWithCall` | 1.5 | `for x: f(x)` |
| `DeepBranch` | 2.0 | `if … if … if …` (≥ 3 deep) |
| `RepeatedComprehension` | 1.0 | Multiple list-comps in one fn |

Lower total cost → higher graphlet score (range `[0, 1]`).

### 3. Runtime Profiling — `environment/track_c.py`
Each candidate refactor is **actually executed** in a timeout-bounded subprocess profiler: CPU time via `time.perf_counter`, peak memory via `tracemalloc`. If a file cannot be safely executed, Track C falls back to compile-time profiling so training never hangs.

### 4. Composable Rubric — `environment/rubrics.py`

Following [OpenEnv RFC 004](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md), the reward is a tree of named, composable child rubrics — modeled on PyTorch's `nn.Module`. When `openenv-core` is installed we extend its `Rubric` base class directly; otherwise we provide a zero-dependency fallback shim with the same surface area.

```
GreenCodeRubric                       (root)
└─ Sequential                         ← short-circuits if any gate fails
   ├─ syntax_gate     {0,1}           ← reward = 0 if any file doesn't parse
   ├─ hack_gate       {0,1}           ← reward = -1 if test files are tampered with
   └─ WeightedSum                     ← soft signals
      ├─ green       (0.70)
      │  ├─ graphlet (0.40) ∈ [0,1]
      │  ├─ cpu      (0.35) ∈ [0,1]
      │  └─ memory   (0.25) ∈ [0,1]
      └─ compliance  (0.30)
```

Each child is independently inspectable via `rubric.named_rubrics()`, so training infrastructure logs every component without modifying the rubric. Adding a new signal (e.g. a Big-O complexity penalty) is a 5-line subclass + `WeightedSum` weight tweak.

The rubric tree is also exposed at runtime via `GET /rubric` so judges can introspect the reward without touching server internals.

```
R = (syntax_gate ∧ hack_gate) × (0.70·green + 0.30·compliance) − P_efficiency
```
- **Why this is hard to game** — the `Sequential` short-circuits any soft reward when the syntax gate fails. Agents can't get points for "memory-efficient" code that doesn't compile.
- **`P_efficiency`** — `0.01` per file edited; held outside the rubric (it's a training-time minimal-edits nudge, not a property of the env).

### 5. CO₂ Dashboard — `environment/co2_calculator.py`
CPU-time savings × CPU TDP × grid carbon intensity → kg CO₂/year, with real-world equivalents (tree-years, car-km). Live HTML dashboard at `/dashboard/co2/{episode_id}`.

---

## 📊 Evidence the Agent Actually Learns

We ran a **25-episode baseline comparison** before any RL training, scoring policies on identical episodes:

| Policy | Mean reward | Green score | Compliance | CO₂ saved/year |
|--------|------------:|------------:|-----------:|---------------:|
| **No-op** (does nothing) | 0.270 | 0.386 | 0.00 | 0.42 kg |
| **Oracle** (cheats — sees the answer) | **0.527** | 0.392 | 0.84 | **0.83 kg** |
| **Trained agent** *(after fast A100 GRPO run)* | _TBD — fill in after run_ | _TBD_ | _TBD_ | _TBD_ |

The **96 % gap between no-op and oracle** proves the env has a strong, learnable signal. Reproduce locally:

```bash
python training/compare_baseline.py --num-episodes 20
# → assets/baseline_vs_trained.png + .json
```

![Baseline vs Trained](assets/baseline_vs_trained.png)

### Training curves

`training/train_grpo.py` writes `assets/training_curves.png` and `assets/log_history.json` after the A100 GRPO run. Commit those files immediately after the final run so judges can verify the real loss/reward curves.

---

## 🏆 Why This Wins (mapped to judging criteria)

| Criterion | Weight | What we deliver |
|-----------|------:|-----------------|
| **Environment Innovation** | 40 % | Graphlet-based control-flow analysis as a learnable structure prior; runtime-grounded reward (not subjective rules); CO₂ dashboard converting reward into real-world impact; curriculum that escalates *corruption intensity*, not just rule count. |
| **Storytelling & Presentation** | 30 % | Live `/demo` page with side-by-side before/after; rich HTML CO₂ dashboard at `/dashboard/co2/{id}`; concrete worked example in this README. |
| **Showing Improvement** | 20 % | Pre-training baseline vs. oracle ceiling **already shipped** (96 % spread proves signal); auto-saved loss + reward curves from `train_grpo.py`; baseline comparison reproducible in one command. |
| **Reward & Pipeline Coherence** | 10 % | Multiplicative test-gate × (0.70 green + 0.30 compliance) − P_eff. Hard gate prevents reward-hacking via broken code. Anti-cheat layer (`P_hack`) penalises tampering with test infrastructure. |

---

## 🔗 Submission Links

| Resource | Link |
|----------|------|
| 🤗 **HF Space** | https://huggingface.co/spaces/s123hree/green-code-optimizer-a100 |
| 🧑‍💻 **Repository** | https://github.com/bcde123/Meta-Round2 |
| 📝 **Blog Post** | https://github.com/bcde123/Meta-Round2/blob/main/blog_post.md |

---

## 🧩 OpenEnv Compatibility

This env follows the [OpenEnv](https://github.com/meta-pytorch/openenv) spec (RFC 001 + RFC 004).

- **Manifest:** [`openenv.yaml`](openenv.yaml)
- **Server:** uses `openenv-core>=0.2.3` for the `Rubric` base class
- **Gym-style API:** `POST /reset`, `POST /step`, `GET /state/{id}`
- **Rubric introspection:** `GET /rubric` returns the named child tree
- **Sync client:** [`client.py`](client.py) — drop-in `EnvClient`, mirrors `HTTPEnvClient`
- **Observation space:** `{ files, violation_report, steps_remaining, curriculum_level }`
- **Action space:** `[read_file, edit_file, run_tests, check_compliance]`
- **Reward range:** `[-1.0, 1.0]`
- **Max episode length:** 70 steps

```python
# Three-line judge-friendly usage:
from client import GreenCodeEnv
env = GreenCodeEnv("https://s123hree-green-code-optimizer-a100.hf.space")
obs = env.reset(curriculum_level=2)        # gym-style reset
state = env.state()                         # gym-style state
print(env.rubric_tree())                    # introspect the reward
```

---

## 📚 API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/demo` | GET | **🌱 Start here** — live before/after demo |
| `/dashboard/co2/{episode_id}` | GET | **CO₂-savings dashboard** (HTML for browsers, JSON otherwise) |
| `/rubric` | GET | **Rubric tree** — named children + formula |
| `/` | GET | Project info |
| `/health` · `/health/green` | GET | Health checks |
| `/docs` | GET | Swagger UI |
| `/reset` | POST | Start a new episode (Gym API) |
| `/step` | POST | Submit an edit (Gym API) |
| `/state/{episode_id}` | GET | Episode metadata (Gym API) |
| `/infer` | POST | Run trained agent (GPU) |

---

## 🚀 Quickstart

### Run the env locally
```bash
git clone https://huggingface.co/spaces/s123hree/green-code-optimizer-a100
cd green-code-optimizer-a100
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
# Visit http://localhost:7860/demo
```

### Reproduce training (A100, deadline-safe)
The Space defaults to a fast final run: `TRAIN_MAX_STEPS=80`, `TRAIN_NUM_GENERATIONS=2`, `TRAIN_NUM_EPISODES=80`, `LORA_RANK=8`, and compile-mode green profiling. Increase these env vars only if you have extra time.

### Reproduce baseline comparison (CPU, ~30 s)
```bash
python training/compare_baseline.py --num-episodes 20
```

### Smoke-test the deployed Space
```bash
SPACE_URL=https://s123hree-green-code-optimizer-a100.hf.space \
  python test_deployment.py
```

---

## 🛠 Repository Layout

```
.
├── server.py                    # FastAPI: /reset /step /state /rubric /demo /dashboard
├── client.py                    # Sync EnvClient — mirrors OpenEnv HTTPEnvClient
├── inference.py                 # Loads adapter from HF Hub, runs predictions
├── openenv.yaml                 # OpenEnv manifest
├── Dockerfile                   # HF Space image
├── blog_post.md                 # Writeup (problem → env → rubric → results)
├── environment/
│   ├── rubrics.py               # ⭐ Composable Rubric tree (RFC 004)
│   ├── episode_generator.py     # Corruption pipeline + curriculum
│   ├── track_a.py               # Code-quality evaluator
│   ├── track_b.py               # Compliance checker
│   ├── track_c.py               # Green-code evaluator (CPU + memory)
│   ├── graphlet_analyzer.py     # Control-flow pattern detection
│   ├── co2_calculator.py        # CPU-time → kg CO₂/year
│   ├── rule_engine.py           # 150-rule cascading rule engine
│   ├── ENGINEERING_STANDARDS.md # The 150 rules
│   └── base_codebase/           # Original clean Python codebase
├── training/
│   ├── train_grpo.py            # GRPO trainer with auto-plotting
│   ├── compare_baseline.py      # Baseline-vs-trained comparison
│   └── verify_pipeline.py       # CPU-only pipeline sanity check
├── notebooks/train_grpo.ipynb   # Colab-ready training notebook
└── assets/
    ├── baseline_vs_trained.png
    ├── baseline_vs_trained.json
    ├── system_architecture_diagram.png
    ├── architecture_pipeline.png
    └── co2_pipeline_diagram.png
```

---

## 🤝 Extending the Env

- **New reward signal** → write a 5-line `Rubric` subclass and add it to the `WeightedSum` in `environment/rubrics.py::build_green_rubric`. Every leaf is auto-logged via `named_rubrics()`.
- **New graphlet patterns** → add to `PATTERN_COSTS` in `environment/graphlet_analyzer.py`.
- **New energy-degrading corruptions** → add a method to `EpisodeGenerator` and append it to `energy_corruptions` list.
- **Tune carbon constants** for your region's grid → `environment/co2_calculator.py`.

```python
# Drop-in custom rubric example:
from environment.rubrics import Rubric, build_green_rubric, WeightedSum

class BigOPenalty(Rubric):
    def forward(self, action, observation) -> float:
        # ... your big-O analysis ...
        return score   # range [0, 1]

base = build_green_rubric()
base.green = WeightedSum(
    [base.green.graphlet, base.green.cpu, base.green.memory, BigOPenalty()],
    weights=[0.30, 0.30, 0.20, 0.20],
)
```

---

## 📜 License

Apache-2.0. Fork it, use it, save some carbon.

---

*Built for the **OpenEnv India Hackathon 2026** — Meta PyTorch.*
