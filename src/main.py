import os

# Set all Hugging Face cache directories BEFORE importing transformers
# This ensures all downloads go to G: drive instead of C: drive
cache_base = r"G:\huggingface_cache"
os.environ["HF_HOME"] = cache_base
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_base, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "models")

from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

device = "cuda" 
print(f"Using device: {device}")

model_name = "gpt2"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Simple text generation
input_text = "The future of artificial intelligence is"
print(f"\nInput: {input_text}")
print("\nGenerated text:")

inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

