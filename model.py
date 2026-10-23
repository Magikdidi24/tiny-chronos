"""Chronos-style tokenizer and a small decoder-only transformer."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# uniform bins over [-LIMIT, LIMIT], after mean scaling
N_BINS = 1024
LIMIT = 15.0


def mean_scale(context):
    return context.abs().mean(-1, keepdim=True).clamp_min(1e-5)


def encode(x, scale):
    # out of range values go to the first/last bin
    edges = torch.linspace(-LIMIT, LIMIT, N_BINS + 1, device=x.device)
    return torch.bucketize(x / scale, edges[1:-1])


def decode(tokens, scale):
    centers = -LIMIT + (tokens + 0.5) * (2 * LIMIT / N_BINS)
    return centers * scale


@dataclass
class ModelConfig:
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_len: int = 320  # context + horizon tokens


class TinyChronos(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(N_BINS, c.dim)
        self.pos_emb = nn.Embedding(c.max_len, c.dim)
        layer = nn.TransformerEncoderLayer(c.dim, c.n_heads, 4 * c.dim, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, c.n_layers, norm=nn.LayerNorm(c.dim), enable_nested_tensor=False)
        self.head = nn.Linear(c.dim, N_BINS, bias=False)

        for name, p in self.named_parameters():
            if p.dim() == 2:
                # gpt-2 style: smaller init for the residual projections
                std = 0.02 / math.sqrt(2 * c.n_layers) if name.endswith(("out_proj.weight", "linear2.weight")) else 0.02
                nn.init.normal_(p, std=std)
            elif name.endswith("bias"):
                # zero biases, otherwise all cloned layers start with the same random mlp biases
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        logits = self.head(self.blocks(x, mask=mask, is_causal=True))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def forecast(self, context, horizon, n_samples=20, temperature=1.0):
        """(B, T) past values -> (B, n_samples, horizon) sampled future values."""
        assert horizon < self.config.max_len, f"horizon must be shorter than max_len={self.config.max_len}"
        context = context[:, -(self.config.max_len - horizon):]
        scale = mean_scale(context).repeat_interleave(n_samples, 0)
        tokens = encode(context.repeat_interleave(n_samples, 0), scale)
        # no kv cache, fine for short horizons
        for _ in range(horizon):
            probs = F.softmax(self(tokens)[0][:, -1] / temperature, -1)
            tokens = torch.cat([tokens, torch.multinomial(probs, 1)], 1)
        return decode(tokens[:, -horizon:], scale).view(len(context), n_samples, horizon)
