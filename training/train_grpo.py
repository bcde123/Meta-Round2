import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"  # Prevent wandb login prompt on Colab
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

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

# ── Unsloth hyperparameters ───────────────────────────────────────────────────
MAX_SEQ_LENGTH = 4096       # Context window for training + generation
LORA_RANK = 32              # LoRA rank (8, 16, 32, 64, 128)
LOAD_IN_4BIT = True         # QLoRA – 4-bit quantization for ~60% VRAM reduction
GPU_MEMORY_UTILIZATION = 0.6  # Fraction of GPU memory for vLLM inference engine

# Auto-detect GPU capabilities:
#   - vLLM requires compute capability >= 8.0 (T4=7.5 crashes)
#   - bf16 requires Ampere+ (compute >= 8.0); T4 must use fp16
def _detect_gpu_caps():
    fast_inference = False
    use_bf16 = False
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:  # A100, L4, H100, etc.
                fast_inference = True
                use_bf16 = True
                print(f"  GPU compute {cc[0]}.{cc[1]} >= 8.0 → vLLM ON, bf16 ON")
            else:
                print(f"  GPU compute {cc[0]}.{cc[1]} < 8.0 (T4/V100) → vLLM OFF, fp16 ON")
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

def _compute_lint_score(code_text: str) -> float:
    issues = 0
    lines = code_text.split("\n")
    for line in lines:
        if len(line) > 88: issues += 1
        if line.rstrip() != line: issues += 1
        if "import *" in line: issues += 1
        if line.strip().startswith("except:"): issues += 1
    return max(0.0, 1.0 - issues / max(len(lines), 1))

def _compute_complexity(files: dict) -> float:
    total_branches = 0
    total_functions = 0
    for content in files.values():
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    for subnode in ast.walk(node):
                        if isinstance(subnode, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                            total_branches += 1
        except SyntaxError:
            pass
    if total_functions == 0: return 0.0
    return total_branches / total_functions

def _compute_module_size_compliance(files: dict) -> float:
    if not files: return 0.0
    compliant = sum(1 for content in files.values() if len(content.split("\n")) <= 200)
    return compliant / len(files)

def _has_docstrings(code_text: str) -> float:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return 0.0
    total = 0
    with_doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total += 1
            if ast.get_docstring(node):
                with_doc += 1
    if total == 0: return 1.0
    return with_doc / total

def _has_type_hints(code_text: str) -> float:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return 0.0
    total_args = 0
    annotated_args = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg not in ("self", "cls"):
                    total_args += 1
                    if arg.annotation is not None:
                        annotated_args += 1
    if total_args == 0: return 1.0
    return annotated_args / total_args

def compute_code_quality_fast(orig_files: dict, updated_files: dict) -> float:
    orig_lint = sum(_compute_lint_score(c) for c in orig_files.values()) / max(len(orig_files), 1)
    new_lint = sum(_compute_lint_score(c) for c in updated_files.values()) / max(len(updated_files), 1)
    lint_improvement = max(0.0, new_lint - orig_lint + 0.5)
    lint_improvement = min(lint_improvement, 1.0)
    orig_complexity = _compute_complexity(orig_files)
    new_complexity = _compute_complexity(updated_files)
    if orig_complexity == 0:
        complexity_score = 1.0
    else:
        complexity_score = max(0.0, (orig_complexity - new_complexity) / orig_complexity)
    size_score = _compute_module_size_compliance(updated_files)
    doc_score = sum(_has_docstrings(c) for c in updated_files.values()) / max(len(updated_files), 1)
    hint_score = sum(_has_type_hints(c) for c in updated_files.values()) / max(len(updated_files), 1)
    total = (0.25 * lint_improvement + 0.20 * complexity_score + 0.20 * size_score + 0.20 * doc_score + 0.15 * hint_score)
    return total

def reward_function(completions, prompts, files, rules_active, **kwargs):
    """Compute rewards for GRPO training.
    
    Uses fast AST-based code quality scoring (no subprocess calls)
    and a graduated format reward to bootstrap learning.
    """
    t0 = time.time()
    rewards = []
    files_batch = [json.loads(f) for f in files]
    rules_batch = [json.loads(r) for r in rules_active]
    
    for idx, (completion, orig_files, active_rules) in enumerate(zip(completions, files_batch, rules_batch)):
        try:
            completion_text = extract_completion_text(completion)
            edits = parse_completions(completion_text)
            format_reward = 0.0
            if "<file" in completion_text: format_reward += 0.05
            if '</file>' in completion_text: format_reward += 0.05
            if edits: format_reward += 0.1 * min(len(edits), 4) / 4.0
            if not edits:
                rewards.append(-0.1 + format_reward)
                continue
            updated_files = orig_files.copy()
            for fname, content in edits.items():
                if fname in updated_files:
                    updated_files[fname] = content
            score_a = compute_code_quality_fast(orig_files, updated_files)
            evaluator_b = ComplianceChecker(STANDARDS_PATH)
            evaluator_b.reset(orig_files, active_rules)
            for fname in edits.keys():
                action = {"tool": "edit_file", "args": {"filename": fname}}
                evaluator_b.step(action, f"Edited {fname}")
            score_b = evaluator_b.get_score()
            reward = 0.6 * score_a + 0.4 * score_b + format_reward
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
    print(f"Loading model via Unsloth FastLanguageModel (vLLM={'ON' if FAST_INFERENCE else 'OFF'})...")
    from_pretrained_kwargs = dict(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )
    if FAST_INFERENCE:
        from_pretrained_kwargs.update(
            fast_inference=True,
            max_lora_rank=LORA_RANK,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
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
        max_steps=100,
        num_generations=4,
        max_completion_length=512,
        max_prompt_length=MAX_SEQ_LENGTH - 512,
        save_steps=25,
        logging_steps=5,
        bf16=USE_BF16,
        fp16=not USE_BF16,
        report_to="none",
    )
    if FAST_INFERENCE:
        grpo_kwargs["use_vllm"] = True
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
