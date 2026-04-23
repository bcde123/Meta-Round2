import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output/final_adapter"))

print("="*60)
print("🔍 Loading Base Model and Tokenizer...")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

print(f"🔗 Attaching trained adapter from: {ADAPTER_PATH}")
if not os.path.exists(ADAPTER_PATH):
    print(f"❌ Error: Adapter not found at {ADAPTER_PATH}")
    print("Are you sure the training finished completely?")
    exit(1)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model ready for inference!\n")

messy_code = """
def calculate(x, y):
    # This function lacks type hints, docstrings, and uses bad variable names
    import os # bad import location
    a = x + y
    if a > 10:
        return a
    else:
        return 0
"""

system_prompt = (
    "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
    "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
    "You must return your edited files using the following exact XML format:\n"
    "<file name=\"filename.py\">\n... complete new code ...\n</file>\n"
    "Do not omit any code inside the file block. Provide the full updated file."
)

user_prompt = f"Here is the codebase to refactor:\n\n--- messy.py ---\n```python\n{messy_code}\n```\n\nPlease refactor and return the updated files."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

print("="*60)
print("🤖 Generating Refactored Code...")
print("="*60)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3, 
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("--- AI OUTPUT ---")
print(generated_text)
print("-----------------")
