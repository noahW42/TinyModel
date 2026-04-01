# train.py
# Train the TinyTransformer on a small text dataset
# Usage: python train.py --data data.txt --save_path tiny_model.pth

import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer
from model import TinyTransformer
import os
import math
from tqdm import tqdm

class LangDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx: idx+self.seq_len]
        y = self.token_ids[idx+1: idx+self.seq_len+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--save_path", type=str, default="tiny_model.pth")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=516)
    args = parser.parse_args()

    print("Loading data...")
    texts = load_texts(args.data)
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts, max_vocab=args.max_vocab)
    tokenizer.save(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

 
    all_tokens = []
    for t in texts:
        ids = tokenizer.encode(t)
        all_tokens.extend(ids + [tokenizer.word2idx.get(".", tokenizer.word2idx[tokenizer.unk_token])])

    print("Creating dataset...")
    dataset = LangDataset(all_tokens, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cpu")
    model = TinyTransformer(vocab_size=tokenizer.vocab_size,
                            d_model=args.d_model,
                            nhead=args.nhead,
                            num_layers=args.num_layers,
                            dim_ff=args.d_model*2,
                            max_seq=args.seq_len)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        it = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # (batch, seq, vocab)
            # shift: logits predict next token at each position; yb already shifted
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            it += 1
            pbar.set_postfix({"loss": f"{total_loss/it:.4f}"})
        avg = total_loss / max(1, it)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

    print("Saving model and tokenizer...")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len
        },
    }, args.save_path)
    print("Done.")

if __name__ == "__main__":
    main()
