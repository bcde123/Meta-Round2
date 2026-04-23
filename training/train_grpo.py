import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"  # Prevent wandb login prompt on Colab
import re
import json
import random
import ast
import torch
import datasets
try:
    import wandb  # TRL 1.2.0 references wandb internally during checkpoint saves
except ImportError:
    wandb = None
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.episode_generator import EpisodeGenerator
from environment.track_b import ComplianceChecker

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

def parse_completions(completion_text):
    """Parse the LLM's output to extract edited files."""
    pattern = r'<file name="(.*?)">(.*?)</file>'
    matches = re.findall(pattern, completion_text, flags=re.DOTALL)
    edits = {}
    for filename, content in matches:
        edits[filename.strip()] = content.strip()
    return edits

def extract_completion_text(completion):
    """Extract raw text from a TRL completion (handles both string and chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL chat format: [{"role": "assistant", "content": "..."}]
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback: concatenate all content
        return " ".join(msg.get("content", "") for msg in completion if isinstance(msg, dict))
    return str(completion)

# ---------------------------------------------------------------------------
# Lightweight code-quality scoring (no subprocess calls)
# ---------------------------------------------------------------------------

def _compute_lint_score(code_text: str) -> float:
    """Heuristic lint score without running ruff subprocess."""
    issues = 0
    lines = code_text.split("\n")
    for line in lines:
        if len(line) > 88:
            issues += 1
        if line.rstrip() != line:
            issues += 1
        if "import *" in line:
            issues += 1
        if line.strip().startswith("except:"):
            issues += 1
    # Normalize: fewer issues = higher score
    return max(0.0, 1.0 - issues / max(len(lines), 1))

def _compute_complexity(files: dict) -> float:
    """Compute average cyclomatic complexity via AST."""
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
    if total_functions == 0:
        return 0.0
    return total_branches / total_functions

def _compute_module_size_compliance(files: dict) -> float:
    """Fraction of files under 200 lines."""
    if not files:
        return 0.0
    compliant = sum(1 for content in files.values() if len(content.split("\n")) <= 200)
    return compliant / len(files)

def _has_docstrings(code_text: str) -> float:
    """Fraction of functions that have docstrings."""
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
    if total == 0:
        return 1.0
    return with_doc / total

def _has_type_hints(code_text: str) -> float:
    """Fraction of function args that have type annotations."""
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
    if total_args == 0:
        return 1.0
    return annotated_args / total_args

def compute_code_quality_fast(orig_files: dict, updated_files: dict) -> float:
    """Fast code-quality scoring using AST analysis only (no subprocess calls).
    
    Returns a score between 0.0 and 1.0.
    Compares updated files against original baselines for lint and complexity,
    and measures absolute quality for docstrings, type hints, and module size.
    """
    # Aggregate lint scores
    orig_lint = sum(_compute_lint_score(c) for c in orig_files.values()) / max(len(orig_files), 1)
    new_lint = sum(_compute_lint_score(c) for c in updated_files.values()) / max(len(updated_files), 1)
    lint_improvement = max(0.0, new_lint - orig_lint + 0.5)  # center at 0.5 baseline
    lint_improvement = min(lint_improvement, 1.0)
    
    # Complexity reduction
    orig_complexity = _compute_complexity(orig_files)
    new_complexity = _compute_complexity(updated_files)
    if orig_complexity == 0:
        complexity_score = 1.0
    else:
        complexity_score = max(0.0, (orig_complexity - new_complexity) / orig_complexity)
    
    # Module size compliance
    size_score = _compute_module_size_compliance(updated_files)
    
    # Docstrings
    doc_score = sum(_has_docstrings(c) for c in updated_files.values()) / max(len(updated_files), 1)
    
    # Type hints
    hint_score = sum(_has_type_hints(c) for c in updated_files.values()) / max(len(updated_files), 1)
    
    total = (0.25 * lint_improvement +
             0.20 * complexity_score +
             0.20 * size_score +
             0.20 * doc_score +
             0.15 * hint_score)
    return total


def reward_function(completions, prompts, files, rules_active, **kwargs):
    """Compute rewards for GRPO training.
    
    Uses fast AST-based code quality scoring (no subprocess calls)
    and a graduated format reward to bootstrap learning.
    """
    rewards = []
    
    files_batch = [json.loads(f) for f in files]
    rules_batch = [json.loads(r) for r in rules_active]
    
    for completion, orig_files, active_rules in zip(completions, files_batch, rules_batch):
        try:
            # Fix #3: Extract text from TRL's completion format
            completion_text = extract_completion_text(completion)
            edits = parse_completions(completion_text)
            
            # Fix #5: Graduated format reward — reward partial progress
            format_reward = 0.0
            if "<file" in completion_text:
                format_reward += 0.05  # attempted the format
            if '</file>' in completion_text:
                format_reward += 0.05  # closed a tag
            if edits:
                format_reward += 0.1 * min(len(edits), 4) / 4.0  # valid file tags
            
            if not edits:
                # Even with no valid edits, reward format attempts
                rewards.append(-0.1 + format_reward)
                continue
            
            # Apply edits to the original files
            updated_files = orig_files.copy()
            for fname, content in edits.items():
                if fname in updated_files:
                    updated_files[fname] = content
                    
            # 1. Track A: Fast code quality (AST-based, no subprocess)
            score_a = compute_code_quality_fast(orig_files, updated_files)
            
            # 2. Track B: Compliance Checker
            evaluator_b = ComplianceChecker(STANDARDS_PATH)
            evaluator_b.reset(orig_files, active_rules)
            
            for fname in edits.keys():
                action = {"tool": "edit_file", "args": {"filename": fname}}
                evaluator_b.step(action, f"Edited {fname}")
                
            score_b = evaluator_b.get_score()
            
            # Final Reward: additive so one zero doesn't kill the signal
            reward = 0.6 * score_a + 0.4 * score_b + format_reward
            
            rewards.append(reward)
            
        except Exception as e:
            print(f"Reward calculation error: {e}")
            rewards.append(-0.1)

    return rewards

def create_training_dataset(num_episodes=50):
    """Generate training episodes with file-boundary-aware truncation."""
    print(f"Generating {num_episodes} episodes for training dataset...")
    generator = EpisodeGenerator(BASE_DIR)
    
    dataset_dict = {
        "prompt": [],
        "files": [],
        "rules_active": []
    }
    
    for _ in range(num_episodes):
        ep = generator.generate()
        
        # Fix #7: Truncate at file boundaries instead of mid-code
        code_context = ""
        for fname, content in ep["files"].items():
            file_block = f"\n--- {fname} ---\n```python\n{content}\n```\n"
            if len(code_context) + len(file_block) > 8000:
                break  # stop adding files instead of cutting mid-code
            code_context += file_block
            
        system_prompt = (
            "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
            "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
            "You must return your edited files using the following exact XML format:\n"
            "<file name=\"filename.py\">\n... complete new code ...\n</file>\n"
            "Do not omit any code inside the file block. Provide the full updated file."
        )
        
        user_prompt = f"Here is the codebase to refactor:\n{code_context}\n\nPlease refactor and return the updated files."
        
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        dataset_dict["prompt"].append(prompt_messages)
        dataset_dict["files"].append(json.dumps(ep["files"]))
        dataset_dict["rules_active"].append(json.dumps(ep["rules_active"]))
        
    return datasets.Dataset.from_dict(dataset_dict)

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # ---- H100 80GB: Native bf16, no quantization needed ----
    # 7B model in bf16 = ~14GB, leaving ~65GB for activations, generations, optimizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Built into PyTorch, no extra install needed
    )
    
    lora_config = LoraConfig(
        r=32,                        # Higher rank = more capacity for learning
        lora_alpha=64,               # 2x rank is a good default
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # All linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output"))
    os.makedirs(output_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=2,    # H100 can handle batch_size=2
        gradient_accumulation_steps=4,    # Effective batch = 2*4 = 8
        max_steps=100,                    # More training steps for real learning
        num_generations=4,                # 4 completions per prompt = better GRPO signal
        generation_batch_size=4,          # Generate all 4 at once (H100 has headroom)
        max_completion_length=1024,       # Full-length completions for complete file output
        save_steps=25,                    # Save checkpoint every 25 steps
        logging_steps=5,                  # Log more frequently
        bf16=True,                        # H100 has native bf16
        report_to="none"
    )
    
    # Generate training dataset — more episodes for better coverage
    train_dataset = create_training_dataset(num_episodes=200)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting GRPO Training...")
    torch.cuda.empty_cache()
    trainer.train()
    
    # Save the final adapter
    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    print(f"✅ Training complete! Adapter saved to {final_path}")

if __name__ == "__main__":
    main()
