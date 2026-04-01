# interface.py
# Simple CLI to compare tiny model and HF fine-tuned model.
# Usage: python interface.py

import argparse
import subprocess
import sys
import os
import torch
from generate import generate, load_tokenizer  # generate uses TinyTransformer
from tokenizer import SimpleTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_hf(prompt, hf_dir="hf_finetuned", max_length=80):
    tok = AutoTokenizer.from_pretrained(hf_dir if os.path.isdir(hf_dir) else "distilgpt2")
    model = AutoModelForCausalLM.from_pretrained(hf_dir if os.path.isdir(hf_dir) else "distilgpt2")
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_length=max_length, do_sample=True, top_k=40, temperature=1.0, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

def generate_tiny(prompt, model_path="tiny_model.pth", tokenizer_path="tokenizer.json"):
    # reuse generate.generate function but load tokenizer directly
    from generate import generate as tiny_generate
    from tokenizer import SimpleTokenizer
    t = SimpleTokenizer()
    t.load(tokenizer_path)
    ckpt = torch.load(model_path, map_location="cpu")
    from model import TinyTransformer
    cfg = ckpt["config"]
    model = TinyTransformer(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"], dim_ff=cfg["d_model"]*2, max_seq=cfg["seq_len"])
    model.load_state_dict(ckpt["model_state"])
    return tiny_generate(model, t, prompt, max_len=80, temperature=1.0, top_k=40)

def main():
    print("Simple compare CLI. Make sure you have trained tiny_model.pth and/or hf_finetuned.")
    prompt = input("Enter prompt: ").strip()
    print("\n--- Tiny model output ---")
    try:
        out = generate_tiny(prompt)
        print(out)
    except Exception as e:
        print("Tiny model error:", e)
    print("\n--- Hugging Face model output ---")
    try:
        out2 = generate_hf(prompt)
        print(out2)
    except Exception as e:
        print("HF model error:", e)

if __name__ == "__main__":
    main()
