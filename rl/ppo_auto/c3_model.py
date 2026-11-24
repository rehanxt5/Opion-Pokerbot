# PPO auto cell C: model def + init from previous iteration (notebook cell 5)
import os

import torch
import torch.nn as nn

N_TOK, N_CTX, N_ACT, VOCAB = 41, 17, 6, 85


def _map_tokens(t):
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


class PPOPolicy(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, f=512):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(N_TOK, d)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(N_CTX), nn.Linear(N_CTX, d))
        self.blocks = nn.ModuleList(Block(d, heads, f) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_ACT)
        self.value = nn.Linear(d, 1)

    def trunk(self, tokens, ctx):
        x = self.emb(_map_tokens(tokens)) + self.pos.weight[None, :N_TOK]
        x = x + self.ctx_proj(ctx.float())[:, None, :]
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x).mean(dim=1)

    def forward(self, tokens, ctx):
        return self.head(self.trunk(tokens, ctx))

    def forward_both(self, tokens, ctx):
        p = self.trunk(tokens, ctx)
        return self.head(p), self.value(p).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


dev = "cuda"
model = PPOPolicy().to(dev)
INIT = ("rl_iter/policy_rl.pt" if os.path.exists("rl_iter/policy_rl.pt")
        else "rl_iter/final_act_v2.pt")
ck = torch.load(INIT, map_location="cpu")
sd = ck["model"] if "model" in ck else ck
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"init from {INIT} | missing={list(missing)} unexpected={list(unexpected)}")
assert not unexpected, "bad checkpoint"
print("params:", f"{model.n_params():,}")
print("CELL_C_DONE", flush=True)
