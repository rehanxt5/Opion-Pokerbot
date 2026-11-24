"""Phase-3 distillation trainer (T4 GPU).

Loss = cross-entropy against the teacher's soft labels (p_soft), masked to
legal actions. This is what makes the student learn the teacher's mixed
strategies (bluff frequencies, raise-size splits), not just its argmax moves.

Usage:
  python3 train.py --data opion/data/packed --out /content/ckpt            # full
  python3 train.py --data opion/data/packed/w1/sb18 --smoke                # smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import OpionPolicy  # noqa: E402
from data import load_shards, to_gpu, metrics  # noqa: E402


def evaluate(model, val, batch=16384):
    model.eval()
    kl_s = tv_s = ag_s = ag_hard = n = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(val["act_idx"]), batch):
            sl = slice(i, i + batch)
            tok = val["tokens"][sl]
            ctx = val["ctx"][sl]
            legal = val["legal"][sl]
            logits = model(tok, ctx).float()  # fp16-safe: cast before masking
            logits = logits.masked_fill(legal == 0, float("-inf"))
            q = torch.softmax(logits.float(), -1)
            p = val["p_soft"][sl].float()
            kl, tv, ag = metrics(p, q)
            b = len(p)
            kl_s += kl * b
            tv_s += tv * b
            ag_s += ag * b
            ag_hard += (q.argmax(-1) == val["act_idx"][sl]).float().sum().item()
            n += b
    model.train()
    return kl_s / n, tv_s / n, ag_s / n, ag_hard / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/content/ckpt")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
    ap.add_argument("--val-cap", type=int, default=200_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.batch, a.max_steps, a.val_cap = 1.0, 512, 300, 20_000

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={dev} "
          f"({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})",
          flush=True)

    t0 = time.time()
    tr_raw, n_tr, n_files = load_shards(a.data, parts=(0,))
    va_raw, n_va, _ = load_shards(a.data, parts=(1,))
    print(f"[train] {n_files} shards | train={n_tr:,} val={n_va:,} "
          f"(loaded {time.time() - t0:.0f}s)", flush=True)
    if n_va > a.val_cap:
        for k in va_raw:
            va_raw[k] = va_raw[k][:a.val_cap]
    tr = to_gpu(tr_raw, dev)
    va = to_gpu(va_raw, dev)
    free, total = torch.cuda.mem_get_info()
    print(f"[train] data on GPU | free={(free >> 20)}MB/{(total >> 20)}MB",
          flush=True)

    model = OpionPolicy().to(dev)
    print(f"[train] params={model.n_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps_per_epoch = n_tr // a.batch
    total_steps = int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler()
    print(f"[train] steps={total_steps} (batch={a.batch}, "
          f"{a.epochs} epochs)", flush=True)

    log = {"steps": [], "val": []}
    t0 = time.time()
    for step in range(1, total_steps + 1):
        idx = torch.randint(0, n_tr, (a.batch,), device=dev)
        tok = tr["tokens"][idx]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(tok, tr["ctx"][idx]).float()  # fp16-safe: cast before masking
            logits = logits.masked_fill(tr["legal"][idx] == 0, -1e9)
            logq = torch.log_softmax(logits, -1)
            loss = -(tr["p_soft"][idx].float() * logq).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % 100 == 0 or step == total_steps:
            with torch.no_grad():
                q = torch.softmax(logits.float(), -1)
                hard_ag = (q.argmax(-1) == tr["act_idx"][idx]).float().mean()
            el = time.time() - t0
            print(f"[train] step {step}/{total_steps} loss={loss.item():.4f} "
                  f"hard_ag={hard_ag.item():.3f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{el:.0f}s ({step / el:.0f} it/s)", flush=True)
        if step % 1000 == 0 or step == total_steps:
            kl, tv, ag, ag_hard = evaluate(model, va)
            log["val"].append({"step": step, "kl": kl, "tv": tv,
                               "ag": ag, "ag_hard": ag_hard})
            log["steps"].append(step)
            print(f"[val] step {step}: KL={kl:.4f} TV={tv:.4f} "
                  f"argmax_ag={ag:.3f} hard_ag={ag_hard:.3f}", flush=True)
            torch.save({"model": model.state_dict(), "step": step},
                       os.path.join(a.out, "last.pt"))

    kl, tv, ag, ag_hard = evaluate(model, va)
    final = {"kl": kl, "tv": tv, "argmax_ag": ag, "hard_ag": ag_hard,
             "steps": total_steps, "train_n": n_tr, "smoke": bool(a.smoke),
             "wall_min": round((time.time() - t0) / 60, 1)}
    torch.save({"model": model.state_dict(), "final": final},
               os.path.join(a.out, "final.pt"))
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump({"log": log, "final": final}, f, indent=1)
    print("[final] " + json.dumps(final), flush=True)
    print("TRAIN_" + ("SMOKE_DONE" if a.smoke else "DONE"), flush=True)


if __name__ == "__main__":
    main()
