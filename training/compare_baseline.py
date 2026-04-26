"""compare_baseline.py — produce the "did the agent actually learn?" plot.

Runs N freshly generated episodes against:
  1. A no-op baseline (agent does nothing — proves the env actually has
     a green-code optimisation opportunity).
  2. A heuristic baseline (apply the inverse of every corruption — the
     ceiling reward an oracle agent would get).
  3. The trained adapter (Qwen-1.5B + LoRA), if HF_ADAPTER_REPO is set.

Saves:
  assets/baseline_vs_trained.png   — bar chart of mean reward
  assets/baseline_vs_trained.json  — raw numbers for the README

Use:
  python training/compare_baseline.py --num-episodes 30 --include-trained false

Designed to be runnable on CPU (skip --include-trained) so judges can
verify the env without a GPU. With a GPU, pass --include-trained true.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.episode_generator import EpisodeGenerator
from environment.track_b import ComplianceChecker
from environment.track_c import GreenCodeEvaluator
from environment.co2_calculator import generate_dashboard_data


BASE_DIR = ROOT / "environment" / "base_codebase"
STANDARDS = ROOT / "environment" / "ENGINEERING_STANDARDS.md"


def _binary_test(files: dict) -> float:
    import ast
    for content in files.values():
        try:
            ast.parse(content)
        except SyntaxError:
            return 0.0
    return 1.0


def _compliance_score(orig: dict, updated: dict, rules_active: list) -> float:
    """Mirrors the compliance computation in training/train_grpo.py."""
    checker = ComplianceChecker(str(STANDARDS))
    checker.reset(orig, rules_active)
    for fname in updated:
        if fname in orig and updated[fname] != orig[fname]:
            checker.step({"tool": "edit_file", "args": {"filename": fname}},
                         f"Edited {fname}")
    return checker.get_score()


def _score(orig: dict, updated: dict, rules_active: list) -> dict:
    """Identical scoring formula to training/train_grpo.py."""
    s_test = _binary_test(updated)
    green = GreenCodeEvaluator().evaluate(orig, updated)
    compliance = _compliance_score(orig, updated, rules_active)

    blended = 0.70 * green.total + 0.30 * compliance
    reward = s_test * blended
    co2 = generate_dashboard_data(green, orig, updated)["co2_savings"]
    return {
        "reward": round(reward, 4),
        "s_test": s_test,
        "green_total": green.total,
        "graphlet": green.graphlet_score,
        "cpu_improvement": green.cpu_improvement,
        "memory_improvement": green.memory_improvement,
        "compliance": round(compliance, 4),
        "co2_kg_per_year": co2["kg_per_year"],
    }


def _noop_baseline(episode):
    """Agent does nothing → submits the corrupted code unchanged."""
    return episode["files"]


def _oracle_baseline(episode, generator):
    """Cheating ceiling: return the original clean files."""
    return generator._load_base_files()


def _trained_agent(episode, model, tokenizer):
    """Run the trained model on the episode and parse XML edits."""
    import re
    import torch

    code_context = ""
    for fname, content in episode["files"].items():
        code_context += f"\n--- {fname} ---\n```python\n{content}\n```\n"

    system = (
        "You are an expert Python refactoring agent. Refactor the codebase "
        "for energy efficiency: replace nested loops, hoist invariants, use "
        "comprehensions. Preserve all logic. Return your edits as XML:\n"
        '<file name="filename.py">\n... full new code ...\n</file>\n'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": code_context},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    edits = {}
    for m in re.finditer(r'<file\s+name="([^"]+)">\s*(.*?)\s*</file>', completion, re.DOTALL):
        edits[m.group(1)] = m.group(2)

    updated = dict(episode["files"])
    updated.update({k: v for k, v in edits.items() if k in updated})
    return updated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--include-trained", default="false",
                    help="true|false — load the trained adapter and score it (requires GPU)")
    ap.add_argument("--adapter-repo", default=os.getenv("HF_ADAPTER_REPO",
                    "shreeyanshi03/constrained-refactor-adapter-1.5b"))
    args = ap.parse_args()
    include_trained = args.include_trained.lower() == "true"

    print(f"📊 Generating {args.num_episodes} episodes for baseline comparison...")
    generator = EpisodeGenerator(str(BASE_DIR))
    episodes = [generator.generate() for _ in range(args.num_episodes)]

    results = {"noop": [], "oracle": []}
    if include_trained:
        results["trained"] = []

    print("\n[1/3] no-op baseline (agent does nothing)...")
    for ep in episodes:
        results["noop"].append(_score(ep["files"], _noop_baseline(ep), ep["rules_active"]))

    print(f"[2/3] oracle baseline (cheats — knows the original)...")
    for ep in episodes:
        results["oracle"].append(
            _score(ep["files"], _oracle_baseline(ep, generator), ep["rules_active"])
        )

    if include_trained:
        print(f"[3/3] trained agent ({args.adapter_repo}) ...")
        from inference import _load_model_once  # type: ignore
        model, tokenizer = _load_model_once()
        for ep in episodes:
            updated = _trained_agent(ep, model, tokenizer)
            results["trained"].append(_score(ep["files"], updated, ep["rules_active"]))

    # ── Aggregate ───────────────────────────────────────────────────────────
    summary = {}
    for name, runs in results.items():
        if not runs:
            continue
        summary[name] = {
            "n": len(runs),
            "mean_reward": round(sum(r["reward"] for r in runs) / len(runs), 4),
            "mean_green": round(sum(r["green_total"] for r in runs) / len(runs), 4),
            "mean_compliance": round(sum(r["compliance"] for r in runs) / len(runs), 4),
            "mean_co2_kg_per_year": round(sum(r["co2_kg_per_year"] for r in runs) / len(runs), 4),
            "test_pass_rate": round(sum(r["s_test"] for r in runs) / len(runs), 4),
        }

    # ── Persist ─────────────────────────────────────────────────────────────
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "baseline_vs_trained.json").write_text(json.dumps(
        {"summary": summary, "raw": results}, indent=2,
    ))
    print(f"\n💾 Saved {assets/'baseline_vs_trained.json'}")

    # ── Plot ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(summary.keys())
        rewards = [summary[k]["mean_reward"] for k in labels]
        greens = [summary[k]["mean_green"] for k in labels]
        compliances = [summary[k]["mean_compliance"] for k in labels]

        x = range(len(labels))
        width = 0.27

        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar([i - width for i in x], rewards, width, label="reward", color="#10b981")
        bars2 = ax.bar(list(x), greens, width, label="green", color="#34d399")
        bars3 = ax.bar([i + width for i in x], compliances, width, label="compliance", color="#a7f3d0")

        for bars in (bars1, bars2, bars3):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels([l.title() for l in labels])
        ax.set_ylabel("Mean score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Baseline vs. Trained Agent — {args.num_episodes} episodes")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out = assets / "baseline_vs_trained.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"📊 Saved {out}")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

    print("\n=== Summary ===")
    for name, stats in summary.items():
        print(f"  {name:8s} reward={stats['mean_reward']:.3f}  green={stats['mean_green']:.3f}  "
              f"compliance={stats['mean_compliance']:.3f}  CO₂/year={stats['mean_co2_kg_per_year']:.2f}kg")


if __name__ == "__main__":
    main()
