import os
import time
import getpass

# Patch getpass.getuser to avoid KeyError: 'getpwuid(): uid not found: 1000'
# in container environments like Hugging Face Spaces.
try:
    getpass.getuser()
except KeyError:
    def dummy_getuser():
        return os.environ.get("USER", "huggingface")
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

# ── Unsloth + GPU imports (deferred for CPU-only testing) ─────────────────────
try:
    import torch
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)          # patch TRL's GRPOTrainer for 2x speed
    from trl import GRPOConfig, GRPOTrainer
    HAS_GPU = True
except (ImportError, NotImplementedError):
    HAS_GPU = False
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.episode_generator import EpisodeGenerator
from environment.track_b import ComplianceChecker
from environment.track_c import GreenCodeEvaluator

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

# ── Unsloth hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LENGTH = 4096       # Context window for training + generation
LORA_RANK = 32              # LoRA rank (8, 16, 32, 64, 128)
LOAD_IN_4BIT = True         # QLoRA – 4-bit quantization for ~60% VRAM reduction
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
TEST_WEIGHT = 1.0                     # Weight for the binary test gate


# ═══════════════════════════════════════════════════════════════════════════════
# BINARY EXECUTION GATE  (S_test)
# ═══════════════════════════════════════════════════════════════════════════════
def _binary_test_score(files: dict) -> float:
    """Binary execution gate: 1.0 if ALL files parse, 0.0 otherwise.

    This is intentionally brutal — a single SyntaxError in any file
    zeroes out the entire reward via the multiplicative formula.
    """
    for content in files.values():
        try:
            ast.parse(content)
        except SyntaxError:
            return 0.0
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# HACK DETECTION
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
# GRANULAR AST COMPLIANCE  (C_i per rule)
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_granular_compliance(orig_files: dict, updated_files: dict,
                                  active_rules: list) -> float:
    """Per-rule compliance score: (1/N) * sum(C_i).

    Each C_i is 1.0 if the corresponding structural constraint is
    satisfied, 0.0 otherwise.  Uses the ComplianceChecker for rule-
    engine evaluation and augments it with direct AST checks for
    naming, forbidden patterns, and structural requirements.
    """
    evaluator_b = ComplianceChecker(STANDARDS_PATH)
    evaluator_b.reset(orig_files, active_rules)

    # Simulate edits so the rule engine can process resolutions
    for fname in updated_files:
        if fname in orig_files and updated_files[fname] != orig_files[fname]:
            action = {"tool": "edit_file", "args": {"filename": fname}}
            evaluator_b.step(action, f"Edited {fname}")

    engine_score = evaluator_b.get_score()  # 0.0 – 1.0

    # ── AST-level structural bonus checks ──────────────────────────────────
    ast_checks_passed = 0
    ast_checks_total = 0

    for content in updated_files.values():
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                ast_checks_total += 1
                # Check: no single-letter function names (excluding standard ones)
                if len(node.name) > 1 or node.name in ("_",):
                    ast_checks_passed += 1

            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                ast_checks_total += 1
                if node.id not in FORBIDDEN_NAMES and len(node.id) > 1:
                    ast_checks_passed += 1

    if ast_checks_total > 0:
        ast_ratio = ast_checks_passed / ast_checks_total
    else:
        ast_ratio = 1.0

    # Blend: 70% rule-engine, 30% direct AST verification
    return 0.70 * engine_score + 0.30 * ast_ratio


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPLICATIVE REWARD FUNCTION
# R = (W_test * S_test) * compliance_score - P_efficiency - P_hack
# ═══════════════════════════════════════════════════════════════════════════════
def reward_function(completions, prompts, files, rules_active, **kwargs):
    """Compute rewards for GRPO training using the multiplicative formulation.

    Formula:
        R_total = (W_test * S_test) × (1/N * Σ C_i) - P_efficiency - P_hack

    Design principles:
      • Binary test gate: if ANY file has a SyntaxError the multiplier is 0.
      • Granular compliance: per-rule AST verification blended with the
        rule-engine score.
      • Efficiency penalty: 0.01 per file edited to encourage minimal edits.
      • Hack penalty: -1.0 and immediate short-circuit for cheating attempts.
      • A small format bonus is preserved to bootstrap early learning of the
        required XML output structure.
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

            # ── BINARY TEST GATE (S_test) ────────────────────────────────────
            s_test = _binary_test_score(updated_files)

            # ── GRANULAR COMPLIANCE (1/N * Σ C_i) ────────────────────────────
            compliance_score = _compute_granular_compliance(
                orig_files, updated_files, active_rules
            )

            # ── EFFICIENCY PENALTY (P_efficiency) ────────────────────────────
            num_files_edited = sum(
                1 for f in edits if f in orig_files
            )
            p_efficiency = EFFICIENCY_PENALTY_PER_EDIT * num_files_edited

            # ── Green-code efficiency (Track C — supplemental signal) ────────
            evaluator_c = GreenCodeEvaluator()
            green_score = evaluator_c.evaluate(orig_files, updated_files)
            score_c = green_score.total
            print(
                f"    [Track C] completion {idx}: green={score_c:.3f} "
                f"(graphlet={green_score.graphlet_score:.3f}, "
                f"cpu={green_score.cpu_improvement:.3f}, "
                f"mem={green_score.memory_improvement:.3f})"
            )

            # ── MULTIPLICATIVE REWARD ─────────────────────────────────────────
            # R = (W_test * S_test) × compliance - P_efficiency
            # Green score is folded into compliance as a 15% blend
            blended_compliance = 0.85 * compliance_score + 0.15 * score_c
            reward = (
                (TEST_WEIGHT * s_test) * blended_compliance
                - p_efficiency
                + format_bonus
            )

            print(
                f"    [Reward] #{idx}: S_test={s_test:.0f} | "
                f"compliance={compliance_score:.3f} | green={score_c:.3f} | "
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
            "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
            "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
            "You must return your edited files using the following exact XML format:\n"
            "<file name=\"filename.py\">\n... complete new code ...\n</file>\n"
            "Do not omit any code inside the file block. Provide the full updated file."
        )
        user_prompt = f"Here is the codebase to refactor:\n{code_context}\n\nPlease refactor and return the updated files."
        prompt_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        dataset_dict["prompt"].append(prompt_messages)
        dataset_dict["files"].append(json.dumps(ep["files"]))
        dataset_dict["rules_active"].append(json.dumps(ep["rules_active"]))
    return datasets.Dataset.from_dict(dataset_dict)

def main():
    if not HAS_GPU:
        print("❌ GPU required for training. Run this on Colab or a machine with NVIDIA/AMD GPU.")
        return
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
        gradient_accumulation_steps=2,  # Reduced from 4 to fit 8 generations
        max_steps=100,
        num_generations=8,              # 8 generations → more reward variance
        max_completion_length=512,
        max_prompt_length=MAX_SEQ_LENGTH - 512,
        temperature=1.0,                # Higher temperature → diverse completions
        save_steps=25,
        logging_steps=5,
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

if __name__ == "__main__":
    main()
