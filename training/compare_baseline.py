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
from environment.track_c import GreenCodeEvaluator
from environment.co2_calculator import generate_dashboard_data
from environment.rubrics import build_green_rubric, CodeAction, CodeObservation


BASE_DIR = ROOT / "environment" / "base_codebase"
STANDARDS = ROOT / "environment" / "ENGINEERING_STANDARDS.md"

# One rubric instance shared across baselines for apples-to-apples comparison.
RUBRIC = build_green_rubric()


def _score(orig: dict, updated: dict, rules_active: list) -> dict:
    """Score via the canonical Green-Code Rubric (same one used in training).

    Pulls per-component scores from `rubric.named_rubrics()` so the breakdown
    stays in sync with the training pipeline — no duplicate scoring logic.
    """
    reward = RUBRIC(
        action=CodeAction(updated_files=updated),
        observation=CodeObservation(
            orig_files=orig, active_rules=rules_active, standards_path=str(STANDARDS),
        ),
    )
    children = dict(RUBRIC.named_rubrics())

    # CO₂ savings — separate from rubric (it's a presentation metric).
    green = GreenCodeEvaluator().evaluate(orig, updated)
    co2 = generate_dashboard_data(green, orig, updated)["co2_savings"]

    return {
        "reward": round(reward, 4),
        "s_test": children["syntax_gate"].last_score or 0.0,
        "green_total": children["green"].last_score or 0.0,
        "graphlet": children["green.graphlet"].last_score or 0.0,
        "cpu_improvement": children["green.cpu"].last_score or 0.0,
        "memory_improvement": children["green.memory"].last_score or 0.0,
        "compliance": round(children["compliance"].last_score or 0.0, 4),
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

    # Use the SAME prompt the agent was trained on so this baseline measures
    # the trained policy faithfully (no prompt-distribution shift).
    from inference import SYSTEM_PROMPT
    user_prompt = (
        "Refactor the following codebase for energy efficiency. "
        f"Preserve all behaviour; just make it cheaper to run.\n{code_context}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
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

        nice_label = {"noop": "No-op\n(unchanged)", "oracle": "Oracle\n(ceiling)",
                      "trained": "Trained\nAgent"}
        labels = list(summary.keys())
        rewards = [summary[k]["mean_reward"] for k in labels]
        greens = [summary[k]["mean_green"] for k in labels]
        compliances = [summary[k]["mean_compliance"] for k in labels]

        x = range(len(labels))
        width = 0.27

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width for i in x], rewards, width,
                       label="Reward (rubric output)", color="#059669", edgecolor="white")
        bars2 = ax.bar(list(x), greens, width,
                       label="Green score = 0.40·graphlet + 0.35·CPU + 0.25·mem",
                       color="#34d399", edgecolor="white")
        bars3 = ax.bar([i + width for i in x], compliances, width,
                       label="Compliance score (engineering rules)",
                       color="#a7f3d0", edgecolor="white")

        for bars in (bars1, bars2, bars3):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(list(x))
        ax.set_xticklabels([nice_label.get(l, l.title()) for l in labels], fontsize=11)
        ax.set_ylabel("Score (range 0.0 – 1.0, higher = better)", fontsize=11)
        ax.set_xlabel("Agent (averaged over fresh episodes)", fontsize=11)
        ax.set_ylim(0, 1.10)
        ax.set_title(
            f"Green-Code Optimizer — Baseline Comparison ({args.num_episodes} episodes)",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Caption with run metadata
        caption = (
            f"Bars: mean over {args.num_episodes} freshly generated episodes. "
            f"Reward = 0.70·green + 0.30·compliance, gated by syntax/hack checks."
        )
        fig.text(0.5, 0.01, caption, ha="center", fontsize=8, style="italic",
                 color="#475569")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        out = assets / "baseline_vs_trained.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
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
