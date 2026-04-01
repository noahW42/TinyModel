
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dims
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x: (batch, seq, d_model)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=2, num_layers=2, dim_ff=256, max_seq=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # initialize
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        mask = self._causal_mask(x.size(1), x.device)  
        emb = self.token_emb(x)  # (batch, seq, d_model)
        out = self.pos_enc(emb)
        for layer in self.layers:
            out = layer(out, attn_mask=mask)
        out = self.ln(out)
        logits = self.head(out)  # (batch, seq, vocab)
        return logits

    def _causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
