
import argparse
import torch
from tokenizer import SimpleTokenizer
from model import TinyTransformer
import json
import random
import math

def load_tokenizer(path):
    t = SimpleTokenizer()
    t.load(path)
    return t

def top_k_logits(logits, k):
    if k == 0:
        return logits
    v, _ = torch.topk(logits, k)
    minv = v[:, -1].unsqueeze(1)
    return torch.where(logits < minv, torch.full_like(logits, -1e10), logits)

def generate(model, tokenizer, prompt, max_len=80, temperature=1.0, top_k=40):
    device = torch.device("cpu")
    model.eval()

    # get index of <UNK> to mask it
    unk_idx = tokenizer.word2idx.get("<UNK>")

    # encode prompt
    ctx = tokenizer.encode(prompt)
    ctx = ctx[-(model.layers[0].attn.embed_dim*2):]  # truncate if too long
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    generated = ctx.copy()

    with torch.no_grad():
        for _ in range(max_len):
            if x.size(1) > model.pos_enc.pe.size(1):
                x = x[:, -model.pos_enc.pe.size(1):]

            logits = model(x)  # (1, seq, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # mask <UNK> token so it cannot be generated
            if unk_idx is not None:
                logits[:, unk_idx] = -1e10

            if top_k > 0:
                logits = top_k_logits(logits, k=top_k)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

            # break on EOS-ish token: we'll treat '.' as break if repeated
            if tokenizer.idx2word.get(next_id, "") == "." and len(generated) > 10:
                if random.random() < 0.5:
                    break

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tiny_model.pth")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    # load tokenizer
    t = SimpleTokenizer()
    t.load(args.tokenizer)
    # load model weights and config
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = ckpt["config"]
    model = TinyTransformer(vocab_size=cfg["vocab_size"],
                            d_model=cfg["d_model"],
                            nhead=cfg["nhead"],
                            num_layers=cfg["num_layers"],
                            dim_ff=cfg["d_model"]*2,
                            max_seq=cfg["seq_len"])
    model.load_state_dict(ckpt["model_state"])
    out = generate(model, t, args.prompt, max_len=args.max_len, temperature=args.temperature, top_k=args.top_k)
    print("\n=== GENERATED ===\n")
    print(out)
    print("\n=================\n")

if __name__ == "__main__":
    main()
