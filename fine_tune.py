import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch

def load_text_dataset(file_path, tokenizer, block_size=128):
    # Use HF's simple TextDataset builder for small tasks
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--model_name", type=str, default="distilgpt2")
    parser.add_argument("--output_dir", type=str, default="hf_finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = load_text_dataset(args.data, tokenizer, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved fine-tuned model to", args.output_dir)

if __name__ == "__main__":
    main()
