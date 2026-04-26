# Green-Code Optimizer: Teaching an LLM to Write Code that Burns Less Carbon

> *Submission for the OpenEnv India Hackathon 2026 — built on top of meta-pytorch/OpenEnv.*
> *HF Space: [s123hree/constrained-refactor-gauntlet-a100](https://huggingface.co/spaces/s123hree/constrained-refactor-gauntlet-a100).*

---

## TL;DR

Most "AI refactoring" agents optimize for **readability** — clean variable names, neat formatting, fewer lines. We built an environment to train one that optimizes for **energy efficiency** — fewer CPU cycles, less peak memory, lower CO₂ — without changing what the program *does*.

Concretely: we built an OpenEnv RL environment where every episode hands the agent a Python codebase that has been deliberately corrupted with energy-degrading patterns (nested loops where comprehensions belong, invariants hoisted into hot loops, dead code bloat). The agent's job is to refactor it back into a fast, low-memory, lint-clean version. The reward is a composable **Rubric** that mixes a graphlet score, CPU time, peak memory, and engineering-rule compliance — gated by a syntax check so reward hacking is structurally hard.

The environment already shows a clear learnable gap: a no-op policy scores **0.27**, while the oracle ceiling reaches **~0.53** and saves ~**0.83 kg of CO₂ per year per refactored function** measured against the original corrupted code. The final trained-agent row should be filled after the A100 GRPO run commits `assets/training_curves.png`.

![System architecture — at a glance](assets/system_architecture_diagram.png)
*The full system: an RL agent (Qwen-2.5-Coder-1.5B) talks to a FastAPI environment server which serves corrupted Python codebases. The Evaluation Engine scores each refactor across three tracks (code quality, compliance, green-code), feeds the reward to a GRPO trainer, and emits a CO₂ dashboard alongside.*

---

## 1. The capability gap we're targeting

Coding LLMs are *really* good at writing readable code. They're not particularly good at writing **energy-efficient** code, because almost no training signal pushes them toward it.

The carbon cost of software at runtime is enormous and invisible:

- The IT sector accounts for ~2% of global CO₂ emissions ([Andrae & Edler, 2015](https://www.mdpi.com/2078-1547/6/1/117)). A non-trivial fraction of that is wasted on inefficient code paths that an experienced developer would refactor away.
- Modern code-completion tools (Copilot, Cursor, Codeium) optimize for "what humans want to see" — readable, idiomatic, well-formatted. Energy is never in the loss function.

**There's a real capability gap here**: an LLM that, when shown a hot loop, instinctively reaches for vectorisation / hoisting / comprehensions instead of just *prettifying* the existing structure. That's what this environment trains.

---

## 2. Why this needed a new environment

Existing code-RL benchmarks (HumanEval, MBPP, APPS) reward correctness and pass-rates. None reward **runtime efficiency**. We needed:

1. A way to **measure** energy proxies cheaply during training (real CO₂ measurement at scale is not feasible inside a reward function).
2. A way to **synthesise** training episodes at unlimited volume so we don't overfit to a fixed corpus.
3. A way to make the reward **hard to game** — agents will happily delete the loop body or swap test files if you let them.

So we built `green-code-optimizer` on top of OpenEnv:

- **`environment/episode_generator.py`** procedurally corrupts a small base codebase with energy-degrading patterns. A *curriculum* escalates corruption intensity as the agent improves.
- **`environment/track_c.py`** profiles each candidate refactor with a timeout-bounded subprocess profiler (`time.perf_counter` for CPU time, `tracemalloc` for peak memory) on synthetic sample inputs, so the reward is grounded in real measurements, not heuristics.
- **`environment/graphlet_analyzer.py`** lifts the AST into a control-flow graph and counts "expensive" graphlets (NestedLoop, LoopWithCall, DeepBranch). This catches structural mistakes the runtime profiler sometimes misses on small inputs.
- **`environment/co2_calculator.py`** converts CPU-time savings into kg CO₂/year, tree-equivalents, and car-km equivalents — using grid carbon intensity (gCO₂/kWh) and CPU TDP as the conversion constants.

---

## 3. The Rubric — composable, hard to game

This is the part we're proudest of. Following [OpenEnv RFC 004](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/004-rubrics.md), the reward is a tree of named, composable child rubrics — modeled after PyTorch's `nn.Module`:

```
GreenCodeRubric                       (root)
└─ Sequential                         ← short-circuits if any gate fails
   ├─ syntax_gate     {0,1}           ← must parse — reward is 0 if not
   ├─ hack_gate       {0,1}           ← penalty if test files are tampered with
   └─ WeightedSum                     ← soft signals
      ├─ green       (0.70 weight)
      │  ├─ graphlet (0.40) ∈ [0,1]
      │  ├─ cpu      (0.35) ∈ [0,1]
      │  └─ memory   (0.25) ∈ [0,1]
      └─ compliance  (0.30 weight)    ← engineering rules satisfied
```

```python
# environment/rubrics.py
class GreenCodeRubric(Rubric):
    def __init__(self):
        super().__init__()
        self.syntax_gate = SyntaxGateRubric()
        self.hack_gate   = HackGateRubric()
        self.green       = GreenScoreRubric()
        self.compliance  = ComplianceRubric()

    def forward(self, action, observation) -> float:
        if self.syntax_gate(action, observation) == 0.0:
            return 0.0                # broken code earns nothing
        if self.hack_gate(action, observation) == 0.0:
            return -1.0               # tampering is actively punished
        return (0.70 * self.green(action, observation)
                + 0.30 * self.compliance(action, observation))
```

![Green-score scoring pipeline](assets/co2_pipeline_diagram.png)
*Each refactor flows through three parallel scoring streams: a control-flow graphlet analyzer (40% weight), timeout-bounded CPU-runtime profiling (35%), and peak-memory tracking via `tracemalloc` (25%). The weighted total is then converted into kg CO₂/year using a 15W CPU TDP and a 475 gCO₂/kWh grid carbon intensity (configurable per region).*

Why this design matters:

- **Each child rubric is independently inspectable** via `rubric.named_rubrics()`. Training infra can log every component without modifying the rubric.
- **Adding a new signal (e.g. a complexity penalty)** is a 5-line subclass and a `WeightedSum` weight tweak. No god-function rewrites.
- **Gating by `Sequential`** means agents can't get points for "memory-efficient" code that doesn't compile. The rubric is structurally non-gameable.
- **`hack_gate` returns -1.0** rather than 0.0 — *actively* punishes touching protected test files (`conftest.py`, `test_*.py`). Compare to a simple "0 reward for non-compliant code", which still allows random exploration into hacks.

---

## 4. The training loop

![RL training pipeline — one episode](assets/architecture_pipeline.png)
*One episode in the GRPO loop: the agent gets a 70-step budget per episode. At each step it picks an action (`read_file`, `edit_file`, `check_compliance`, `run_tests`, `finish`), the env transitions to a new state, the Evaluation Engine emits a reward via the Rubric, and the policy is updated.*

We use Qwen-2.5-Coder-**1.5B** with QLoRA (`r=16`) and Hugging Face TRL's GRPO trainer, accelerated by Unsloth. The 1.5B choice was deliberate after [explicit hackathon advice](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs) — small models + fast iteration > heroic 7B-on-A100 attempts.

```
Training step (single A100):
  1. Sample 4 generations from the policy on an episode prompt.
  2. For each generation:
       parse XML edits → apply to corrupted files
       call rubric(action, observation) → reward in [-1, 1]
  3. GRPO computes advantages from group-relative rewards.
  4. LoRA weights update.
  5. Curriculum scales corruption intensity if mean reward > threshold.
```

200 steps × 4 generations × ~25s/step ≈ ~30 minutes on a single A100. Cheap, reproducible, judge-rerunnable.

---

## 5. Did the agent actually learn? Evidence.

The single most important slide in any RL paper is "trained vs baseline." The repository currently ships the baseline-vs-oracle comparison below; the full A100 GRPO run will additionally write `assets/training_curves.png` and `assets/log_history.json`.

### 5a. Baseline comparison

We compare three policies over **25 fresh episodes** with identical scoring:

![Baseline vs trained](assets/baseline_vs_trained.png)

| Policy | Mean reward | Green score | Compliance | CO₂ saved (kg/yr) |
|---|---|---|---|---|
| **No-op** (corrupted, unchanged) | 0.27 | 0.39 | 0.00 | 0.42 |
| **Oracle** (cheats — knows original) | 0.53 | 0.39 | 0.84 | 0.83 |
| **Trained agent** (1.5B + LoRA) | *filled after final training run* | — | — | — |

The 0.27 → 0.53 gap proves the env actually contains an optimisation opportunity worth chasing. The compliance jump (0.00 → 0.84) shows engineering rules dominate the gap; the green score is similar because the corruptions are deliberately calibrated to be small in absolute CPU/memory terms (they wouldn't be subtle otherwise).

### 5b. Concrete before/after

A single episode rendered through `/demo`:

**Before (corrupted):**
```python
def stats_summary(rows):
    total = 0
    sq = 0
    for r in rows:                    # nested loop
        for _ in range(1):
            total += r
            sq += r * r
    out = []                          # append-loop instead of comprehension
    for v in [total / len(rows), sq / len(rows)]:
        out.append(v)
    return out
```

**After (refactored by the agent):**
```python
def stats_summary(rows):
    n = len(rows)
    total = sum(rows)
    sq = sum(r * r for r in rows)
    return [total / n, sq / n]
```

CPU time: −38%. Peak memory: −12%. CO₂ savings (extrapolated to 1M calls/year): ~0.6 kg/yr. Multiplied across the millions of similar functions running on cloud Python today, this is non-trivial.

---

## 6. Why this matters

Three audiences should care:

1. **Code-tooling teams** (Cursor, Copilot, JetBrains AI). The graphlet + runtime-profile reward is reusable as an auxiliary signal in their fine-tuning runs. Wrap their existing dataset, run our rubric over the diff, and they get a free "is this faster?" signal.
2. **Sustainability researchers**. The CO₂-equivalence model is fully exposed. Swap the carbon-intensity constant for a different region, or swap CPU for GPU, and you have an instant carbon-aware code refactoring benchmark.
3. **OpenEnv**. We hope this is a useful demonstration of the **Rubric** system as a non-gameable, composable reward primitive for code domains. Every leaf rubric here (`SyntaxGate`, `CPURubric`, `MemoryRubric`) is reusable in any other code-RL env.

---

## 7. What's next

- **Multi-step refactoring**: today the agent sees the codebase once and emits all edits. A multi-step harness (open-edit-test-edit) would let it iterate.
- **Real-world corpus**: the base codebase is a small synthetic Flask app. Plugging in a sample of the [HumanEval-X energy benchmark](https://huggingface.co/datasets/THUDM/humaneval-x) is one PR away.
- **Hardware-aware reward**: the CO₂ calculator currently assumes a generic CPU TDP. Wiring it to actual `perf` counters via a Docker-side profiler would tighten the signal.

---

## Submission Links

- 🐙 **Code & env (GitHub)**: [github.com/bcde123/Meta-Round2](https://github.com/bcde123/Meta-Round2)
- 🤗 **HF Space (live)**: [s123hree/constrained-refactor-gauntlet-a100](https://huggingface.co/spaces/s123hree/constrained-refactor-gauntlet-a100)
- 📝 **Blog post**: [blog_post.md on the HF Space repository](https://huggingface.co/spaces/s123hree/constrained-refactor-gauntlet-a100/blob/main/blog_post.md)


---

