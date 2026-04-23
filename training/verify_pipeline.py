import os, sys, time, json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("=" * 60)
print("TEST 1: GPU Check")
print("=" * 60)
import torch
if not torch.cuda.is_available():
    print("FAIL: No CUDA"); sys.exit(1)
gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU: {gpu_name}, VRAM: {vram:.1f}GB, BF16: {torch.cuda.is_bf16_supported()}")
print("\n" + "=" * 60)
print("TEST 2: Episode Generation")
print("=" * 60)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))
from environment.episode_generator import EpisodeGenerator
gen = EpisodeGenerator(BASE_DIR)
ep = gen.generate()
print(f"  Files: {list(ep['files'].keys())}")
print(f"  Rules: {len(ep['rules_active'])} active")
print("\n" + "=" * 60)
print("TEST 3: Reward Function")
print("=" * 60)
from training.train_grpo import reward_function
bad = reward_function(["no xml here"], ["t"], [json.dumps(ep['files'])], [json.dumps(ep['rules_active'])])
print(f"  Bad format reward:  {bad[0]:.3f} (expect ~-0.1)")
fname = list(ep['files'].keys())[0]
good = f'<file name="{fname}">"""Docstring."""\n{ep["files"][fname]}</file>'
good_r = reward_function([good], ["t"], [json.dumps(ep['files'])], [json.dumps(ep['rules_active'])])
print(f"  Good format reward: {good_r[0]:.3f} (expect >0.3)")
print("\n" + "=" * 60)
print("TEST 4: Model Load + Generate")
print("=" * 60)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa")
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
print(f"  Model loaded in {time.time()-t0:.1f}s")
model.print_trainable_parameters()
inputs = tokenizer("Refactor: def f(x,y): return x+y", return_tensors="pt").to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
gen_time = time.time() - t0
txt = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"  Generated {len(out[0])-inputs['input_ids'].shape[1]} tokens in {gen_time:.1f}s")
print(f"  Output: {txt[:200]}")
print("\n" + "=" * 60)
print("TEST 5: Quick 2-step GRPO Training")
print("=" * 60)
from trl import GRPOConfig, GRPOTrainer
from training.train_grpo import create_training_dataset
train_dataset = create_training_dataset(num_episodes=10)
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output_verify"))
os.makedirs(output_dir, exist_ok=True)
args = GRPOConfig(output_dir=output_dir, learning_rate=1e-5, per_device_train_batch_size=1, gradient_accumulation_steps=1, max_steps=2, num_generations=2, generation_batch_size=2, max_completion_length=128, save_steps=999, logging_steps=1, bf16=True, report_to="none")
trainer = GRPOTrainer(model=model, reward_funcs=reward_function, args=args, train_dataset=train_dataset, processing_class=tokenizer)
print("  Starting 2-step training...")
t0 = time.time()
torch.cuda.empty_cache()
trainer.train()
train_time = time.time() - t0
mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n  2 steps done in {train_time:.1f}s ({train_time/2:.1f}s/step)")
print(f"  Peak GPU memory: {mem:.1f}/{vram:.1f}GB")
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print(f"  Estimated full run (100 steps): ~{train_time/2*100/60:.0f} minutes")
print(f"  Now run: python training/train_grpo.py")
