"""Opion student model v1: ~1M-param transformer policy (PyTorch, T4).

Trunk: token embeddings (cards + action-history) + context projection,
learned slot positions, 4 pre-LN transformer layers (d=128, 4 heads, FFN 512).
Head: 6-bucket policy (elitev6 action space) with legal-action masking.

spec.py (opion/fleet/spec.py) defines the features; the shipped numpy engine
will run the SAME architecture in pure numpy — parity guaranteed by keeping
this file the single training-side definition.
"""
import torch
import torch.nn as nn

N_TOK = 41          # token slots per act decision (see spec.py)
N_CTX = 17          # context dims
N_ACT = 6           # elitev6 buckets: fold/call/small/mid/big/all-in
VOCAB = 85          # 53 card ids (0=pad) + 32 history tokens


def _map_tokens(t):
    """cards stay 0..52; history tokens 1..32 shift to 53..84."""
    t = t.long()
    return torch.where(t < 53, t, 52 + t.clamp(min=1, max=32))


class Block(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.h = h
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, f)
        self.fc2 = nn.Linear(f, d)
        self.act = nn.GELU()

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5), -1)
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x


class OpionPolicy(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, f=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(N_TOK, d)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(N_CTX),
                                      nn.Linear(N_CTX, d))
        self.blocks = nn.ModuleList(Block(d, heads, f) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_ACT)

    def forward(self, tokens, ctx):
        """tokens: [B,41] uint8/long, ctx: [B,17] float -> logits [B,6]."""
        x = self.emb(_map_tokens(tokens)) + self.pos.weight[None, :N_TOK]
        x = x + self.ctx_proj(ctx.float())[:, None, :]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x).mean(dim=1)
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
