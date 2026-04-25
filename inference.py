"""
inference.py — Inference endpoint for the Constrained Refactor Gauntlet agent.

Supports two modes:
  1. Local: loads adapter from grpo_output/final_adapter (with Unsloth)
  2. Docker/HF Spaces: downloads adapter from HuggingFace Hub (vanilla transformers+peft)

Environment variables:
  HF_ADAPTER_REPO  – HuggingFace repo ID for the adapter (e.g. "username/adapter-name")
  HF_TOKEN          – HuggingFace token for private repos
  ADAPTER_LOCAL_PATH – Override local path to adapter directory
"""

import os
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 4096

# Adapter source: HuggingFace Hub repo OR local path
HF_ADAPTER_REPO = os.getenv("HF_ADAPTER_REPO", "")
ADAPTER_LOCAL_PATH = os.getenv(
    "ADAPTER_LOCAL_PATH",
    os.path.join(os.path.dirname(__file__), "grpo_output", "final_adapter")
)
HF_TOKEN = os.getenv("HF_TOKEN", None)

SYSTEM_PROMPT = (
    "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
    "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
    "You must return your edited files using the following exact XML format:\n"
    '<file name="filename.py">\n... complete new code ...\n</file>\n'
    "Do not omit any code inside the file block. Provide the full updated file."
)


def _load_model():
    """Load the base model + LoRA adapter. Works with or without Unsloth."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = torch.cuda.is_available()

    # ── Try Unsloth first (faster, used during training) ─────────────────
    try:
        from unsloth import FastLanguageModel

        adapter_path = ADAPTER_LOCAL_PATH
        if HF_ADAPTER_REPO:
            adapter_path = HF_ADAPTER_REPO

        print(f"Loading via Unsloth: {adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=use_4bit,
        )
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Model loaded via Unsloth")
        return model, tokenizer

    except (ImportError, Exception) as e:
        print(f"Unsloth unavailable ({e}), falling back to transformers+peft...")

    # ── Fallback: vanilla transformers + peft (Docker/CPU) ───────────────
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
    )

    # Load adapter from HF Hub or local path
    adapter_source = HF_ADAPTER_REPO if HF_ADAPTER_REPO else ADAPTER_LOCAL_PATH
    if os.path.isdir(adapter_source) or HF_ADAPTER_REPO:
        print(f"Loading LoRA adapter: {adapter_source}")
        model = PeftModel.from_pretrained(model, adapter_source, token=HF_TOKEN)
        model = model.merge_and_unload()
        print("✅ LoRA adapter merged")
    else:
        print(f"⚠️  No adapter found at {adapter_source}, using base model")

    model.eval()
    print("✅ Model loaded via transformers+peft")
    return model, tokenizer


# ── Singleton model loading ──────────────────────────────────────────────────
_model = None
_tokenizer = None


def get_model():
    """Lazy-load the model on first call."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = _load_model()
    return _model, _tokenizer


def run_inference(observation: dict) -> dict:
    """
    Given an environment observation, generate the next agent action.
    
    For the OpenEnv step-by-step interface (server.py):
      observation = {files, violation_report, steps_remaining, ...}
    
    Returns a tool-call dict: {"tool": "...", "args": {...}}
    """
    model, tokenizer = get_model()

    steps_remaining = observation.get("steps_remaining", 0)
    report = observation.get("violation_report", {})
    files = observation.get("files", {})

    # Build context from files
    code_context = ""
    for fname, content in files.items():
        block = f"\n--- {fname} ---\n```python\n{content}\n```\n"
        if len(code_context) + len(block) > 8000:
            break
        code_context += block

    user_prompt = (
        f"Steps remaining: {steps_remaining}\n"
        f"Violation report: {json.dumps(report, indent=2)}\n"
        f"\nHere is the codebase to refactor:\n{code_context}\n"
        f"\nPlease refactor and return the updated files."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    # Parse the generated XML into edit actions
    import re
    file_pattern = r'<file name="(.*?)">(.*?)</file>'
    matches = re.findall(file_pattern, generated, flags=re.DOTALL)

    if matches:
        # Return the first edit as an action
        fname, content = matches[0]
        return {
            "tool": "edit_file",
            "args": {"filename": fname.strip(), "content": content.strip()},
        }

    # Fallback: check compliance
    return {"tool": "check_compliance", "args": {}}


# ── CLI entrypoint for quick testing ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Loading Model...")
    print("=" * 60)

    model, tokenizer = get_model()

    messy_code = '''
def calculate(x, y):
    import os
    a = x + y
    if a > 10:
        return a
    else:
        return 0
'''

    user_prompt = (
        f"Here is the codebase to refactor:\n\n"
        f"--- messy.py ---\n```python\n{messy_code}\n```\n\n"
        f"Please refactor and return the updated files."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    print("\n" + "=" * 60)
    print("🤖 Generating Refactored Code...")
    print("=" * 60)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("--- AI OUTPUT ---")
    print(generated_text)
    print("-----------------")
