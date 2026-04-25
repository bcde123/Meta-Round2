# ============================================================
# 🚀 Constrained Refactor Gauntlet — Colab Verification Notebook
# ============================================================
#
# HOW TO USE:
#   1. Open Google Colab (https://colab.research.google.com)
#   2. Runtime → Change runtime type → GPU (T4 is free, A100 if Pro)
#   3. Copy-paste this entire file into a Colab cell, or upload as .py
#   4. Run each cell in order
#
# This notebook tests everything in 5 steps:
#   Cell 1: Install dependencies + clone repo
#   Cell 2: Test environment (no GPU)
#   Cell 3: Test reward function (no GPU)
#   Cell 4: Test model loading (GPU)
#   Cell 5: Test 2-step GRPO training (GPU)
# ============================================================


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1: Setup — Install dependencies & clone repo          ║
# ╚══════════════════════════════════════════════════════════════╝

# --- Run this cell first ---
# !pip install unsloth vllm
# !pip install astor ruff trl peft datasets accelerate bitsandbytes triton
# !git clone https://github.com/shreeyanshi123/Meta-Pytorch-huggingFace-round2.git
# !cd Meta-Pytorch-huggingFace-round2 && git checkout feature/rl-training-improvements

# After running the above, restart runtime if prompted, then continue.


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Test Environment & Episode Generation (no GPU)      ║
# ╚══════════════════════════════════════════════════════════════╝

import os, sys
os.chdir("/content/Meta-Pytorch-huggingFace-round2")
sys.path.insert(0, ".")

print("=" * 60)
print("TEST 1: Episode Generation")
print("=" * 60)

from environment.episode_generator import EpisodeGenerator

gen = EpisodeGenerator("environment/base_codebase")
ep = gen.generate()

print(f"  ✅ Files: {list(ep['files'].keys())}")
print(f"  ✅ Rules active: {len(ep['rules_active'])}")
print(f"  ✅ Curriculum level: {ep['curriculum_level']}")
print(f"  ✅ Episode ID: {ep['episode_id'][:8]}...")

# Quick sanity: files should be corrupted
for fname, content in ep['files'].items():
    print(f"     {fname}: {len(content.split(chr(10)))} lines")

print("\n✅ Environment works!")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3: Test Reward Function (no GPU)                       ║
# ╚══════════════════════════════════════════════════════════════╝

import json

print("=" * 60)
print("TEST 2: Reward Function")
print("=" * 60)

from training.train_grpo import reward_function

# 2a: Bad completion
bad = reward_function(
    completions=["no xml here"],
    prompts=["test"],
    files=[json.dumps(ep['files'])],
    rules_active=[json.dumps(ep['rules_active'])]
)
print(f"  Bad format reward:  {bad[0]:.3f}  (expect ~-0.1)")

# 2b: Good completion
fname = list(ep['files'].keys())[0]
good_text = f'<file name="{fname}">"""Docstring."""\n{ep["files"][fname]}</file>'
good = reward_function(
    completions=[good_text],
    prompts=["test"],
    files=[json.dumps(ep['files'])],
    rules_active=[json.dumps(ep['rules_active'])]
)
print(f"  Good format reward: {good[0]:.3f}  (expect >0.3)")

assert bad[0] < 0.1, f"Bad reward too high: {bad[0]}"
assert good[0] > 0.1, f"Good reward too low: {good[0]}"
print("\n✅ Reward function works!")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4: Test Model Loading via Unsloth (GPU required)       ║
# ╚══════════════════════════════════════════════════════════════╝

import time
import torch

print("=" * 60)
print("TEST 3: GPU + Model Loading")
print("=" * 60)

assert torch.cuda.is_available(), "❌ No GPU! Go to Runtime → Change runtime type → GPU"
gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  ✅ GPU: {gpu_name} ({vram:.1f} GB)")

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_RANK = 32
MAX_SEQ_LENGTH = 4096

# Auto-detect GPU capabilities (vLLM disabled due to v0.19.1 BitsAndBytes bug)
cc = torch.cuda.get_device_capability(0)
FAST_INFERENCE = False  # vLLM v0.19.1 crashes with BitsAndBytes
USE_BF16 = cc[0] >= 8
print(f"  {torch.cuda.get_device_name(0)} (compute {cc[0]}.{cc[1]}) → {'bf16' if USE_BF16 else 'fp16'}, vLLM OFF")

t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
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
inputs = tokenizer("Refactor: def f(x,y): return x+y", return_tensors="pt").to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7,
                         pad_token_id=tokenizer.pad_token_id)
gen_time = time.time() - t0
txt = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"  ✅ Generated {len(out[0])-inputs['input_ids'].shape[1]} tokens in {gen_time:.1f}s")
print(f"  Output: {txt[:150]}...")
print("\n✅ Model loads and generates!")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5: Test 2-step GRPO Training (GPU required)            ║
# ╚══════════════════════════════════════════════════════════════╝

print("=" * 60)
print("TEST 4: 2-step GRPO Training")
print("=" * 60)

from trl import GRPOConfig, GRPOTrainer
from training.train_grpo import create_training_dataset

train_dataset = create_training_dataset(num_episodes=10)
print(f"  ✅ Dataset: {len(train_dataset)} episodes")

output_dir = "/content/grpo_output_verify"
os.makedirs(output_dir, exist_ok=True)

grpo_kwargs = dict(
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
    bf16=USE_BF16,
    fp16=not USE_BF16,
    report_to="none",
)
training_args = GRPOConfig(**grpo_kwargs)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

print("  Starting 2-step training...")
t0 = time.time()
torch.cuda.empty_cache()
trainer.train()
train_time = time.time() - t0

mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n  ✅ 2 steps in {train_time:.1f}s ({train_time/2:.1f}s/step)")
print(f"  ✅ Peak GPU: {mem:.1f}/{vram:.1f} GB")

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print(f"""
  GPU:            {gpu_name} ({vram:.0f}GB)
  Model:          {MODEL_NAME} (Unsloth 4-bit)
  Load time:      {load_time:.0f}s
  Training speed: ~{train_time/2:.0f}s/step
  Peak VRAM:      {mem:.1f}GB

  Estimated full training (100 steps): ~{train_time/2*100/60:.0f} minutes

  To run full training:
    python training/train_grpo.py
""")
