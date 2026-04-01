# TinyModel
LLM comparison between huggingface pretrained model vs custom
# Learning Language Model (AI)

A next-token prediction system that compares a custom-built Transformer model against a fine-tuned HuggingFace model. Both models are trained on **The Art of War** by Sun Tzu — making for surprisingly philosophical text generation.

## Models

### TinyTransformer (Custom Built)
A Transformer architecture built from scratch in PyTorch for next-token prediction. Trained entirely on the local dataset with a custom word-level tokenizer.

### HuggingFace Fine-Tuned (distilGPT-2)
A pretrained distilGPT-2 model fine-tuned on the same dataset using the HuggingFace Transformers library. Allows direct comparison between a from-scratch model and a fine-tuned pretrained model.

## Features

- Custom word-level tokenizer with vocabulary filtering and unknown-token handling
- Transformer architecture implemented from scratch in PyTorch
- Fine-tuning pipeline for pretrained HuggingFace models
- Training pipeline with batching, loss tracking, and model checkpointing
- Side by side comparison CLI — run both models on the same prompt and compare outputs

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- tqdm

## Prerequisites

- Python 3.9+
- Anaconda (recommended)

## Installation
```bash
pip install torch transformers tqdm
```

## Usage

### Step 1 — Train the TinyTransformer
```bash
python train.py
```

Optional arguments:
```bash
python train.py --epochs 8 --d_model 128 --lr 0.001
```

### Step 2 — Fine-tune the HuggingFace model
```bash
python fine_tune.py
```

Optional arguments:
```bash
python fine_tune.py --epochs 3 --model_name distilgpt2
```

### Step 3 — Run the comparison interface
```bash
python interface.py
```

Example:
```
Enter prompt: The supreme art of war is
```

Output:
```
--- Tiny model output ---
The supreme art of war is to subdue the enemy without fighting

--- Hugging Face model output ---
The supreme art of war is to know when to attack and when to retreat
```

## Project Structure
```
TinyModel/
├── model.py          # TinyTransformer architecture
├── tokenizer.py      # Custom word-level tokenizer
├── train.py          # Training pipeline for TinyTransformer
├── fine_tune.py      # Fine-tuning pipeline for HuggingFace model
├── generate.py       # Text generation logic
├── interface.py      # CLI to compare both models
├── tokenizer.json    # Saved tokenizer vocabulary
├── data.txt          # Training data (The Art of War)
└── hf_finetuned/     # Saved HuggingFace model config files
```

## Training Data

The models are trained on **The Art of War** by Sun Tzu (`data.txt`). You can swap this out for any plain text file to train on different data — just replace `data.txt` and re-run the training steps.

## Results

The two models produce noticeably different outputs on the same prompt. The HuggingFace fine-tuned model generally produces more coherent text due to the pretrained weights, while the TinyTransformer shows the raw capability of a model trained entirely from scratch on a small dataset.
