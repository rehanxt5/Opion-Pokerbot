"""Data loader for the packed Phase-2 shards (acts only; disc is a later run).

- globs opion/data/packed/{w1,w2}/sb*/acts.npz
- splits by the seed-level `part` field baked in at pack time:
  part 0 = train, 1 = val, 2 = test (test is NEVER loaded here)
- loads everything into RAM, then moves to GPU once (the whole dataset fits
  in T4 memory as uint8/float16 -> zero loader stalls, GPU stays maxed)
"""
import glob
import json

import numpy as np
import torch


def load_shards(root, parts=(0, 1)):
    files = sorted(glob.glob(f"{root}/*/acts.npz"))
    if not files:
        # allow the packed dir itself (single-shard smoke runs)
        files = sorted(glob.glob(f"{root}/acts.npz"))
    if not files:
        raise FileNotFoundError(f"no acts.npz under {root}")
    cols = {}
    n = 0
    seeds = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        sel = np.isin(z["part"], parts)
        for k in ("tokens", "ctx", "legal", "p_soft", "act_idx", "part",
                  "street"):
            a = z[k][sel]
            cols.setdefault(k, []).append(a)
        n += int(sel.sum())
    out = {k: np.concatenate(v) for k, v in cols.items()}
    return out, n, len(files)


def to_gpu(data, device="cuda"):
    t = {
        "tokens": torch.from_numpy(data["tokens"].astype(np.int64)),
        "ctx": torch.from_numpy(data["ctx"].astype(np.float32)),
        "legal": torch.from_numpy(data["legal"].astype(np.float32)),
        "p_soft": torch.from_numpy(data["p_soft"].astype(np.float32)),
        "act_idx": torch.from_numpy(data["act_idx"].astype(np.int64)),
        "street": torch.from_numpy(data["street"].astype(np.int64)),
    }
    return {k: v.to(device, non_blocking=True) for k, v in t.items()}


def metrics(p, q):
    """soft-label fidelity: KL(p||q), total variation, argmax agreement."""
    eps = 1e-8
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    agree = (p.argmax(-1) == q.argmax(-1)).float()
    return kl.mean().item(), tv.mean().item(), agree.mean().item()
