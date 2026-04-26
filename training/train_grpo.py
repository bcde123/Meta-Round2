import os
import pwd
import getpass
import time

# Hard patch pwd.getpwuid to never raise KeyError for the current user
def dummy_getpwuid(uid):
    return ('huggingface', 'x', uid, 1000, 'HuggingFace user', '/home/huggingface', '/bin/sh')

pwd.getpwuid = dummy_getpwuid

# Hard patch getpass.getuser to immediately return the dummy user
def dummy_getuser():
    return "huggingface"

getpass.getuser = dummy_getuser

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor"
os.environ["USER"] = "huggingface"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["LOGNAME"] = "huggingface"
import re
import json
import random
import ast
import datasets
try:
    import wandb
except ImportError:
    wandb = None

import torch

# ── Unsloth + GPU imports ─────────────────────────────────────────────────────
# Imported at module level for GPU capability detection; training imports
# (GRPOConfig, GRPOTrainer) are deferred into main() so any error is visible.
try:
    from unsloth import FastLanguageModel, PatchFastRL
    _UNSLOTH_OK = True
except Exception as e:
    print(f"⚠️  Unsloth import failed: {e}")
    _UNSLOTH_OK = False
# ─────────────────────────────────────────────────────────────────────────────

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.episode_generator import EpisodeGenerator
from environment.rubrics import (
    build_green_rubric,
    CodeAction,
    CodeObservation,
    OPENENV_AVAILABLE,
)

print(f"  Rubric ready (openenv-core integration: {OPENENV_AVAILABLE})")

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

# ── Unsloth hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048       # Smaller window → faster rollouts on 1.5B model
LORA_RANK = 16              # Lower rank sufficient for 1.5B; faster convergence
LOAD_IN_4BIT = True         # QLoRA – keeps VRAM low for more generations per step
GPU_MEMORY_UTILIZATION = 0.6  # Fraction of GPU memory for vLLM inference engine

# Auto-detect GPU capabilities:
#   - bf16 requires Ampere+ (compute >= 8.0); T4 must use fp16
#   - vLLM is DISABLED: v0.19.1 has a graph compilation bug with BitsAndBytes
#     ("Tried to erase Node size_3") that crashes on ALL GPUs (T4, A100, H100).
#     Unsloth's training speedups still work; only generation rollouts fall back
#     to HuggingFace generate() which is slightly slower but reliable.
def _detect_gpu_caps():
    fast_inference = False  # vLLM disabled due to v0.19.1 bug
    use_bf16 = False
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            if cc[0] >= 8:
                use_bf16 = True
                print(f"  {gpu_name} (compute {cc[0]}.{cc[1]}) → bf16 ON, vLLM OFF (v0.19.1 bug)")
            else:
                print(f"  {gpu_name} (compute {cc[0]}.{cc[1]}) → fp16 ON, vLLM OFF")
    except Exception:
        pass
    return fast_inference, use_bf16

FAST_INFERENCE, USE_BF16 = _detect_gpu_caps()
# ─────────────────────────────────────────────────────────────────────────────


def parse_completions(completion_text):
    pattern = r'<file name="(.*?)">(.*?)</file>'
    matches = re.findall(pattern, completion_text, flags=re.DOTALL)
    edits = {}
    for filename, content in matches:
        edits[filename.strip()] = content.strip()
    return edits

def extract_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(msg.get("content", "") for msg in completion if isinstance(msg, dict))
    return str(completion)

# ═══════════════════════════════════════════════════════════════════════════════
# ANTI-CHEAT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PROTECTED_FILES = frozenset({"conftest.py", "test_main.py", "test_api.py",
                             "test_utils.py", "pytest.ini", "setup.cfg"})
FORBIDDEN_NAMES = frozenset({"varelunixo", "xhackbypass", "__reward_stub__"})
EFFICIENCY_PENALTY_PER_EDIT = 0.01   # Small cost per edited file
HACK_PENALTY = -1.0                   # Immediate termination-level penalty


# ═══════════════════════════════════════════════════════════════════════════════
# HACK DETECTION
# Outer layer of defense — detects more reward-hacking patterns than the
# rubric's `HackGateRubric` (which only checks protected-file edits). The
# rubric's gate remains as a defense-in-depth fallback.
# ═══════════════════════════════════════════════════════════════════════════════
def _detect_hack(edits: dict, orig_files: dict) -> tuple:
    """Layered anti-cheat verification.

    Returns (is_hack: bool, reason: str).
    Checks:
      1. Editing protected / test infrastructure files.
      2. Introducing test stubs (e.g. `def test_all(): return True`).
      3. Using forbidden naming conventions.
      4. Deleting all test assertions.
    """
    for fname in edits:
        # 1. Protected-file guard
        basename = fname.rsplit("/", 1)[-1] if "/" in fname else fname
        if basename in PROTECTED_FILES:
            return True, f"Edited protected file: {fname}"
        if basename.startswith("test_") and fname not in orig_files:
            return True, f"Created new test file to bypass checks: {fname}"

    for fname, content in edits.items():
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue  # unparseable code is penalised by binary gate, not hack

        for node in ast.walk(tree):
            # 2. Test-stub detection: test functions that only `return True`
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                body = [n for n in node.body
                        if not isinstance(n, (ast.Expr,))
                        or not isinstance(getattr(n, 'value', None), ast.Constant)]
                if len(body) == 1 and isinstance(body[0], ast.Return):
                    val = body[0].value
                    if isinstance(val, ast.Constant) and val.value is True:
                        return True, f"Test stub detected in {fname}: {node.name}"

            # 3. Forbidden names
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                return True, f"Forbidden name '{node.id}' in {fname}"
            if isinstance(node, ast.FunctionDef) and node.name in FORBIDDEN_NAMES:
                return True, f"Forbidden function name '{node.name}' in {fname}"

        # 4. Deleted all assertions from a file that originally had them
        if fname in orig_files:
            orig_asserts = sum(1 for n in ast.walk(ast.parse(orig_files[fname]))
                               if isinstance(n, ast.Assert))
            new_asserts = sum(1 for n in ast.walk(tree)
                              if isinstance(n, ast.Assert))
            if orig_asserts > 0 and new_asserts == 0:
                return True, f"All assertions deleted from {fname}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSABLE-RUBRIC REWARD FUNCTION
# R = (syntax_gate ∧ hack_gate) × (0.70·green + 0.30·compliance) − P_efficiency
# Implemented in environment/rubrics.py; per-component scores are exposed via
# `rubric.named_rubrics()` for logging.
# ═══════════════════════════════════════════════════════════════════════════════
def reward_function(completions, prompts, files, rules_active, **kwargs):
    """Compute rewards for GRPO training via the composable Green-Code Rubric.

    Formula (via `environment/rubrics.py::build_green_rubric`):
        R = (syntax_gate ∧ hack_gate) × (0.70·green + 0.30·compliance)
            − P_efficiency + format_bonus

    Design principles:
      • Hack outer-layer (`_detect_hack`) returns -1.0 immediately on cheating;
        the rubric's own `hack_gate` is a defense-in-depth fallback.
      • The rubric's `syntax_gate` zeroes the reward if any file fails to parse.
      • `green` decomposes into graphlet (0.40), CPU (0.35), memory (0.25).
      • `compliance` is the engineering-rule satisfaction fraction.
      • `P_efficiency` (0.01 per edit) sits OUTSIDE the rubric — it's a
        training-time minimal-edits nudge, not part of the env reward.
      • `format_bonus` bootstraps early learning of the required XML structure.
    """
    t0 = time.time()
    rewards = []
    files_batch = [json.loads(f) for f in files]
    rules_batch = [json.loads(r) for r in rules_active]

    for idx, (completion, orig_files, active_rules) in enumerate(
        zip(completions, files_batch, rules_batch)
    ):
        try:
            completion_text = extract_completion_text(completion)
            edits = parse_completions(completion_text)

            # ── Format bonus (bootstrap signal) ──────────────────────────────
            format_bonus = 0.0
            if "<file" in completion_text:
                format_bonus += 0.02
            if "</file>" in completion_text:
                format_bonus += 0.02
            if edits:
                valid_edits = {k: v for k, v in edits.items() if k in orig_files}
                format_bonus += 0.06 * min(len(valid_edits), 4) / 4.0

            if not edits:
                rewards.append(-0.1 + format_bonus)
                continue

            # ── Apply edits ──────────────────────────────────────────────────
            updated_files = orig_files.copy()
            for fname, content in edits.items():
                if fname in updated_files:
                    updated_files[fname] = content

            # ── HACK DETECTION (P_hack) ──────────────────────────────────────
            is_hack, hack_reason = _detect_hack(edits, orig_files)
            if is_hack:
                print(f"    ⚠ HACK detected (completion {idx}): {hack_reason}")
                rewards.append(HACK_PENALTY)
                continue

            # ── EFFICIENCY PENALTY (P_efficiency) ────────────────────────────
            # Held outside the rubric: it's a training-time minimal-edits
            # nudge, not a property of the environment's reward function.
            num_files_edited = sum(
                1 for f in edits if f in orig_files
            )
            p_efficiency = EFFICIENCY_PENALTY_PER_EDIT * num_files_edited

            # ── COMPOSABLE RUBRIC SCORE ──────────────────────────────────────
            # The Rubric does:
            #   syntax_gate ∧ hack_gate × (0.70·green + 0.30·compliance)
            # where green = 0.40·graphlet + 0.35·cpu + 0.25·memory.
            # All component scores are exposed via `named_rubrics()`.
            # Fresh rubric per completion keeps `last_score` logs accurate even
            # when a gate short-circuits and downstream children are skipped.
            rubric = build_green_rubric()
            rubric_score = rubric(
                action=CodeAction(updated_files=updated_files),
                observation=CodeObservation(
                    orig_files=orig_files,
                    active_rules=active_rules,
                    standards_path=STANDARDS_PATH,
                ),
            )

            # Pull individual component scores for logging
            children = dict(rubric.named_rubrics())
            s_test = children["syntax_gate"].last_score or 0.0
            green = children["green"].last_score or 0.0
            graphlet = children["green.graphlet"].last_score or 0.0
            cpu = children["green.cpu"].last_score or 0.0
            mem = children["green.memory"].last_score or 0.0
            compliance_score = children["compliance"].last_score or 0.0

            # Format bonus only applies after the rubric gives a positive score.
            # Broken or empty code must not earn points for merely using XML.
            reward = rubric_score - p_efficiency
            if rubric_score > 0:
                reward += format_bonus

            print(
                f"    [Reward] #{idx}: gate={s_test:.0f} | "
                f"green={green:.3f} (g={graphlet:.2f}, cpu={cpu:.2f}, mem={mem:.2f}) | "
                f"compliance={compliance_score:.3f} | "
                f"P_eff={p_efficiency:.3f} | R={reward:.4f}"
            )
            rewards.append(reward)

        except Exception as e:
            print(f"Reward calculation error: {e}")
            rewards.append(-0.1)

    elapsed = time.time() - t0
    avg_r = sum(rewards) / len(rewards) if rewards else 0
    print(f"  [Reward] {len(rewards)} completions scored in {elapsed:.1f}s | avg={avg_r:.3f}")
    return rewards

def create_training_dataset(num_episodes=50):
    print(f"Generating {num_episodes} episodes for training dataset...")
    generator = EpisodeGenerator(BASE_DIR)
    dataset_dict = {"prompt": [], "files": [], "rules_active": []}
    for _ in range(num_episodes):
        ep = generator.generate()
        code_context = ""
        for fname, content in ep["files"].items():
            file_block = f"\n--- {fname} ---\n```python\n{content}\n```\n"
            if len(code_context) + len(file_block) > 8000: break
            code_context += file_block
        system_prompt = (
            "You are an expert Python refactoring agent focused on ENERGY EFFICIENCY.\n"
            "Your goal: minimise CPU cycles and peak memory while preserving program logic.\n"
            "Specifically prefer:\n"
            "  • List/dict/set comprehensions over append-loops\n"
            "  • Vectorised / built-in operations (sum, map) over manual accumulation\n"
            "  • Hoisting loop-invariant work outside the loop\n"
            "  • Eliminating dead code and redundant computation\n"
            "  • Flattening unnecessarily nested loops\n"
            "Do NOT alter test files or break any existing assertions.\n"
            "Return edited files using EXACTLY this XML format:\n"
            "<file name=\"filename.py\">\n... complete new code ...\n</file>\n"
            "Provide the full updated file content (do not omit any code)."
        )
        user_prompt = (
            f"Refactor the following codebase for energy efficiency. "
            f"Preserve all behaviour; just make it cheaper to run.\n{code_context}"
        )
        prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        dataset_dict["prompt"].append(prompt_messages)
        dataset_dict["files"].append(json.dumps(ep["files"]))
        dataset_dict["rules_active"].append(json.dumps(ep["rules_active"]))
    return datasets.Dataset.from_dict(dataset_dict)

def main():
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected. Training requires an NVIDIA GPU.")
        return
    if not _UNSLOTH_OK:
        print("❌ Unsloth failed to import — cannot train. Check logs above for the error.")
        return

    # Defer these imports to here so errors are visible instead of silently caught
    try:
        PatchFastRL("GRPO", FastLanguageModel)
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        print(f"❌ Failed to patch/import GRPO trainer: {e}")
        raise

    # ── 1. Load model via Unsloth (replaces manual transformers + peft setup) ──
    print("Loading model via Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Add LoRA adapters via Unsloth (optimised kernels + memory savings) ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0,          # Unsloth recommends 0 – optimised for it
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's long-context checkpointing
        random_state=3407,
    )
    model.print_trainable_parameters()

    # ── 3. Training config ─────────────────────────────────────────────────────
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output"))
    os.makedirs(output_dir, exist_ok=True)
    grpo_kwargs = dict(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=200,                  # More steps → better policy on small model
        num_generations=4,              # 4 gens per step; faster iteration
        max_completion_length=512,
        max_prompt_length=MAX_SEQ_LENGTH - 512,
        temperature=0.9,
        save_steps=50,
        logging_steps=10,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to="none",
    )
    training_args = GRPOConfig(**grpo_kwargs)

    # ── 4. Train ───────────────────────────────────────────────────────────────
    train_dataset = create_training_dataset(num_episodes=200)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    print("Starting GRPO Training (Unsloth + vLLM)...")
    torch.cuda.empty_cache()
    trainer.train()

    # ── 5. Save adapter ───────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Training complete! Adapter saved to {final_path}")

    # ── 5b. Save loss + reward plots (evidence for hackathon submission) ──────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history = trainer.state.log_history or []
        steps = [h.get("step") for h in history if "loss" in h or "reward" in h]
        losses = [h.get("loss") for h in history if "loss" in h]
        rewards = [h.get("reward") for h in history if "reward" in h]

        # Persist raw history for reproducibility
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
        os.makedirs(assets_dir, exist_ok=True)
        with open(os.path.join(assets_dir, "log_history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

        # X-axis = global trainer step (not row index) when available
        loss_pts = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
        rew_pts = [(h["step"], h["reward"]) for h in history if "reward" in h and "step" in h]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle(
            f"Green-Code Optimizer — GRPO training run "
            f"(Qwen-2.5-Coder-1.5B, LoRA r={LORA_RANK}, "
            f"reward = 0.70·green + 0.30·compliance)",
            fontsize=11, fontweight="bold", y=1.02,
        )
        if loss_pts:
            xs, ys = zip(*loss_pts)
            axes[0].plot(xs, ys, color="#dc2626", linewidth=1.8, marker="o", markersize=3)
            axes[0].set_title("Policy loss (lower is better)", fontsize=11)
            axes[0].set_xlabel("Training step", fontsize=10)
            axes[0].set_ylabel("Loss (cross-entropy, nats)", fontsize=10)
            axes[0].grid(True, alpha=0.3, linestyle="--")
            axes[0].set_axisbelow(True)
        if rew_pts:
            xs, ys = zip(*rew_pts)
            axes[1].plot(xs, ys, color="#059669", linewidth=1.8, marker="o", markersize=3,
                         label="Mean reward / batch")
            # Reference lines: no-op and oracle baselines from compare_baseline.py
            try:
                bl_path = os.path.join(assets_dir, "baseline_vs_trained.json")
                if os.path.exists(bl_path):
                    with open(bl_path) as f:
                        bl = json.load(f)["summary"]
                    if "noop" in bl:
                        axes[1].axhline(bl["noop"]["mean_reward"], color="#94a3b8",
                                        linestyle=":", linewidth=1.5,
                                        label=f"No-op baseline ({bl['noop']['mean_reward']:.2f})")
                    if "oracle" in bl:
                        axes[1].axhline(bl["oracle"]["mean_reward"], color="#3b82f6",
                                        linestyle="--", linewidth=1.5,
                                        label=f"Oracle ceiling ({bl['oracle']['mean_reward']:.2f})")
            except Exception:
                pass
            axes[1].set_title("Episode reward (higher is better)", fontsize=11)
            axes[1].set_xlabel("Training step", fontsize=10)
            axes[1].set_ylabel("Reward (rubric, range −1.0 to 1.0)", fontsize=10)
            axes[1].grid(True, alpha=0.3, linestyle="--")
            axes[1].set_axisbelow(True)
            axes[1].legend(loc="lower right", fontsize=8, framealpha=0.95)
        plt.tight_layout()
        plot_path = os.path.join(assets_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Saved training plots to {plot_path}")
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

    # ── 6. Upload adapter to HF Hub (so it survives container restarts) ───────
    hub_repo = os.getenv("HF_ADAPTER_REPO", "shreeyanshi03/constrained-refactor-adapter-1.5b")
    hf_token = os.getenv("HF_TOKEN", None)
    if hub_repo and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True, private=False)
            api.upload_folder(
                folder_path=final_path,
                repo_id=hub_repo,
                repo_type="model",
                commit_message="GRPO training run complete — adapter upload",
            )
            assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
            if os.path.isdir(assets_dir):
                api.upload_folder(
                    folder_path=assets_dir,
                    path_in_repo="assets",
                    repo_id=hub_repo,
                    repo_type="model",
                    commit_message="upload training plots",
                )
            print(f"✅ Adapter + plots uploaded to https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"⚠️  Hub upload failed (adapter still saved locally): {e}")
    else:
        print("ℹ️  HF_TOKEN or HF_ADAPTER_REPO not set — skipping Hub upload")

if __name__ == "__main__":
    main()
