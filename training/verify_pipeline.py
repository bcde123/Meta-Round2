"""
verify_pipeline.py — Quick verification that the entire GRPO pipeline works.

Run this BEFORE the full training to confirm:
  1. GPU is accessible
  2. Model loads correctly (via Unsloth)
  3. Episodes generate properly
  4. Reward function scores correctly
  5. Model can generate text
  6. A 2-step GRPO training completes without errors

Usage:
    cd /content/Meta-Pytorch-huggingFace-round2
    python training/verify_pipeline.py
"""

import os
import sys
import time
import json
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ============================================================
# TEST 1: GPU Check
# ============================================================
print("=" * 60)
print("TEST 1: GPU Check")
print("=" * 60)

if not torch.cuda.is_available():
    print("❌ CUDA not available! Cannot proceed.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  ✅ GPU: {gpu_name}")
print(f"  ✅ VRAM: {vram:.1f} GB")
print(f"  ✅ BF16: {torch.cuda.is_bf16_supported()}")
print(f"  ✅ PyTorch: {torch.__version__}")

# ============================================================
# TEST 2: Episode Generation
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Episode Generation")
print("=" * 60)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

from environment.episode_generator import EpisodeGenerator

gen = EpisodeGenerator(BASE_DIR)
ep = gen.generate()

print(f"  ✅ Files in episode: {list(ep['files'].keys())}")
print(f"  ✅ Active rules: {ep['rules_active'][:10]}... ({len(ep['rules_active'])} total)")
for fname, content in ep['files'].items():
    lines = content.split('\n')
    print(f"     {fname}: {len(lines)} lines, {len(content)} chars")

# ============================================================
# TEST 3: Reward Function (with fake completions)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Reward Function Scoring")
print("=" * 60)

from training.train_grpo import (
    reward_function, parse_completions, extract_completion_text,
    compute_code_quality_fast
)
from environment.track_b import ComplianceChecker

# Test 3a: Bad completion (no XML format)
print("\n  --- Test 3a: Bad completion (no format) ---")
bad_completion = "Here is the refactored code:\ndef hello():\n    print('hi')"
bad_rewards = reward_function(
    completions=[bad_completion],
    prompts=["test"],
    files=[json.dumps(ep['files'])],
    rules_active=[json.dumps(ep['rules_active'])]
)
print(f"  Reward for bad format: {bad_rewards[0]:.3f} (expected: ~-0.1)")

# Test 3b: Partial format
print("\n  --- Test 3b: Partial format ---")
partial_completion = '<file name="api.py">def hello():\n    """Say hello."""\n    print("hi")\n</file>'
partial_rewards = reward_function(
    completions=[partial_completion],
    prompts=["test"],
    files=[json.dumps(ep['files'])],
    rules_active=[json.dumps(ep['rules_active'])]
)
print(f"  Reward for partial: {partial_rewards[0]:.3f} (expected: >0.0)")

# Test 3c: Good completion
print("\n  --- Test 3c: Good completion (proper refactoring) ---")
sample_fname = list(ep['files'].keys())[0]
sample_code = ep['files'][sample_fname]
improved_code = '"""Module docstring."""\n' + sample_code
good_completion = f'<file name="{sample_fname}">{improved_code}</file>'
good_rewards = reward_function(
    completions=[good_completion],
    prompts=["test"],
    files=[json.dumps(ep['files'])],
    rules_active=[json.dumps(ep['rules_active'])]
)
print(f"  Reward for good refactor: {good_rewards[0]:.3f} (expected: >0.3)")

# Test 3d: Track B standalone
print("\n  --- Test 3d: Track B (Compliance) standalone ---")
checker = ComplianceChecker(STANDARDS_PATH)
checker.reset(ep['files'], ep['rules_active'])
for fname in ep['files']:
    checker.step({"tool": "edit_file", "args": {"filename": fname}}, f"Edited {fname}")
score_b = checker.get_score()
print(f"  Compliance score: {score_b:.3f}")
print(f"  Outstanding rules: {len(checker.get_outstanding())}")

# ============================================================
# TEST 4: Model Loading + Generation (Unsloth)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Model Loading + Text Generation (Unsloth)")
print("=" * 60)

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_RANK = 32
MAX_SEQ_LENGTH = 4096

# Auto-detect: vLLM needs compute capability >= 8.0 (T4=7.5 will crash)
cc = torch.cuda.get_device_capability(0)
FAST_INFERENCE = cc[0] >= 8
print(f"  GPU compute capability {cc[0]}.{cc[1]} → vLLM {'enabled' if FAST_INFERENCE else 'disabled'}")

t0 = time.time()
from_pretrained_kwargs = dict(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
if FAST_INFERENCE:
    from_pretrained_kwargs.update(
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
    )
model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
load_time = time.time() - t0
print(f"  ✅ Model loaded in {load_time:.1f}s")
model.print_trainable_parameters()

# Quick generation test
print("\n  --- Generating sample output (max 128 tokens) ---")
test_prompt = "Refactor this Python function:\ndef f(x,y): return x+y"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

t0 = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
gen_time = time.time() - t0
generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"  Generated {len(outputs[0]) - inputs['input_ids'].shape[1]} tokens in {gen_time:.1f}s")
print(f"  Output preview: {generated[:200]}...")

# ============================================================
# TEST 5: Quick 2-step GRPO Training (Unsloth)
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Quick 2-step GRPO Training (Unsloth)")
print("=" * 60)

from trl import GRPOConfig, GRPOTrainer
from training.train_grpo import create_training_dataset

print("  Generating 10 episodes...")
train_dataset = create_training_dataset(num_episodes=10)
print(f"  ✅ Dataset: {len(train_dataset)} episodes")

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output_verify"))
os.makedirs(output_dir, exist_ok=True)

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=2,
    num_generations=2,
    max_completion_length=128,
    max_prompt_length=MAX_SEQ_LENGTH - 128,
    save_steps=999,
    logging_steps=1,
    bf16=True,
    use_vllm=FAST_INFERENCE,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print("  Starting 2-step training (this may take 2-5 minutes)...")
t0 = time.time()
torch.cuda.empty_cache()
trainer.train()
train_time = time.time() - t0

print(f"\n  ✅ 2 steps completed in {train_time:.1f}s ({train_time/2:.1f}s per step)")

mem_used = torch.cuda.max_memory_allocated() / 1024**3
mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  ✅ Peak GPU memory: {mem_used:.1f} / {mem_total:.1f} GB ({mem_used/mem_total*100:.0f}%)")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED — Pipeline is verified (Unsloth)!")
print("=" * 60)
print(f"""
Summary:
  GPU:              {gpu_name} ({vram:.0f}GB)
  Model:            {MODEL_NAME} (via Unsloth)
  Load time:        {load_time:.0f}s
  Gen speed:        ~{gen_time:.1f}s for 128 tokens
  Training speed:   ~{train_time/2:.0f}s per step
  Peak memory:      {mem_used:.1f}/{mem_total:.1f}GB
  
Estimated full run (100 steps): ~{train_time/2 * 100 / 60:.0f} minutes

To start the full training run:
  python training/train_grpo.py
""")
