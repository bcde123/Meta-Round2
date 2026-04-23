import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import wandb
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
BASE_DIR = os.path.join(os.path.dirname(__file__), "../environment/base_codebase")
STANDARDS_PATH = os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md")

def parse_completions(completion_text):
    """Parse the LLM's output to extract edited files."""
    pattern = r'<file name="(.*?)">(.*?)</file>'
    matches = re.findall(pattern, completion_text, flags=re.DOTALL)
    edits = {}
    for filename, content in matches:
        edits[filename.strip()] = content.strip()
    return edits

def reward_function(completions, prompts, files, rules_active, **kwargs):
    rewards = []
    
    # kwargs will contain stringified JSON arrays for files and rules_active from the dataset
    files_batch = [json.loads(f) for f in files]
    rules_batch = [json.loads(r) for r in rules_active]
    
    for completion, orig_files, active_rules in zip(completions, files_batch, rules_batch):
        try:
            edits = parse_completions(completion)
            
            # If model produced no valid edits, give a small negative signal
            # rather than 0.0 (which causes zero-variance and no learning)
            if not edits:
                rewards.append(-0.1)
                continue
            
            # Apply edits to the original files
            updated_files = orig_files.copy()
            for fname, content in edits.items():
                if fname in updated_files:
                    updated_files[fname] = content
                    
            # 1. Track A: Code Quality Evaluator
            evaluator_a = CodeQualityEvaluator()
            evaluator_a.evaluate(orig_files)  # sets the baseline
            score_a = evaluator_a.evaluate(updated_files)  # gets the improved score
            
            # 2. Track B: Compliance Checker
            evaluator_b = ComplianceChecker(STANDARDS_PATH)
            evaluator_b.reset(orig_files, active_rules)
            
            # Simulate steps for each edit
            for fname in edits.keys():
                action = {"tool": "edit_file", "args": {"filename": fname}}
                evaluator_b.step(action, f"Edited {fname}")
                
            score_b = evaluator_b.get_score()
            
            # Final Reward: use additive combination so one zero doesn't kill the signal
            # Track A weighted 0.6, Track B weighted 0.4
            reward = 0.6 * score_a.total + 0.4 * score_b
            
            # Small format bonus: model produced valid <file> tags
            reward += 0.1 * min(len(edits), 4) / 4.0
            
            rewards.append(reward)
            
            if wandb.run:
                wandb.log({
                    "reward_mean": reward,
                    "CodeScore": score_a.total,
                    "ComplianceScore": score_b,
                    "NumEdits": len(edits)
                })
        except Exception as e:
            print(f"Reward calculation error: {e}")
            rewards.append(-0.1)

    return rewards

def create_training_dataset(num_episodes=50):
    print(f"Generating {num_episodes} episodes for training dataset...")
    generator = EpisodeGenerator(BASE_DIR)
    
    dataset_dict = {
        "prompt": [],
        "files": [],
        "rules_active": []
    }
    
    for _ in range(num_episodes):
        ep = generator.generate()
        
        # Format the files into a readable string
        code_context = ""
        for fname, content in ep["files"].items():
            code_context += f"\n--- {fname} ---\n```python\n{content}\n```\n"
            
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
        
    # Colab GPU support: detect architecture to set dtype and use 4-bit quantization
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quant_config,
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=25,
        num_generations=2,           # Minimum for GRPO (need >1 for relative ranking)
        generation_batch_size=2,     # Must be a multiple of num_generations! (Cannot be 1 if num_generations is 2)
        max_completion_length=512,   # Reduced to 512 to fit T4 15GB VRAM
        max_prompt_length=512,       # Cap prompt length to prevent OOM on long code contexts
        save_steps=100,
        logging_steps=10,
        bf16=is_bf16_supported,      # Auto-detect bf16 support
        fp16=not is_bf16_supported,  # Fallback to fp16 for older GPUs (e.g., Colab T4)
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none"
    )
    
    # Generate actual training dataset with real episodes
    train_dataset = create_training_dataset(num_episodes=50)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting GRPO Training...")
    torch.cuda.empty_cache()  # Free any leftover GPU memory before training
    trainer.train()

if __name__ == "__main__":
    main()
