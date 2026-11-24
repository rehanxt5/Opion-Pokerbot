# PPO auto cell G: PPO update (notebook cell 9)
import time

import numpy as np

EPOCHS, MB, CLIP, LR = 2, 4096, 0.2, 2e-5
ENT, VCOEF = 0.01, 0.5
N = len(D["bucket"])
T = torch.from_numpy(D["tokens"].astype(np.int64)).cuda()
C = torch.from_numpy(D["ctx"].astype(np.float32)).cuda()
Lg = torch.from_numpy(D["legal"].astype(np.float32)).cuda()
Bo = torch.from_numpy(D["bucket"].astype(np.int64)).cuda()
Lo = torch.from_numpy(D["logp"].astype(np.float32)).cuda()
Ad = torch.from_numpy(adv.astype(np.float32)).cuda()
Rt = torch.from_numpy(ret.astype(np.float32)).cuda()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")
idx_all = torch.randperm(N, device=dev)
t0 = time.time()
step = 0
for ep in range(EPOCHS):
    for i in range(0, N, MB):
        mb = idx_all[i:i + MB]
        with torch.autocast("cuda", dtype=torch.float16):
            logits, v = model.forward_both(T[mb], C[mb])
        logits = logits.float().masked_fill(Lg[mb] == 0, -1e9)
        logp = torch.log_softmax(logits, -1)
        logp_b = logp.gather(1, Bo[mb, None]).squeeze(1)
        ratio = torch.exp(logp_b - Lo[mb])
        s1 = ratio * Ad[mb]
        s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * Ad[mb]
        pl = -torch.min(s1, s2).mean()
        vl = 0.5 * ((v.float() - Rt[mb]) ** 2).mean()
        q = torch.softmax(logits, -1)
        ent = -(q * torch.nan_to_num(logp, neginf=0.0)).sum(-1).mean()
        ps = slice((step * 4096) % (len(PT) - 4096),
                   (step * 4096) % (len(PT) - 4096) + 4096)
        with torch.autocast("cuda", dtype=torch.float16):
            plogits = model(PT[ps], PC[ps])
        pq = torch.log_softmax(plogits.float(), -1)
        kl = (PP[ps] * (torch.log(PP[ps] + 1e-8) - pq)).sum(-1).mean()
        loss = pl + VCOEF * vl - ENT * ent + BETA * kl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % 10 == 0:
            print(f"  step {step}/{((N + MB - 1) // MB) * EPOCHS} "
                  f"pl={pl.item():+.4f} v={vl.item():.4f} "
                  f"kl={kl.item():.4f} {time.time() - t0:.0f}s", flush=True)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    plogits = model(PT, PC).float()
pq = torch.log_softmax(plogits, -1)
kl_probe = float((PP * (torch.log(PP + 1e-8) - pq)).sum(-1).mean())
pmix = np.bincount(plogits.argmax(-1).cpu().numpy(), minlength=6) / len(PT)
print(f"post-update probe KL(p||pi)={kl_probe:.4f}")
print("probe argmax mix:", np.round(pmix, 3).tolist())
print("teacher ref      : [0.17, 0.44, 0.10, 0.26, 0.04, 0.00]")
print("CELL_G_DONE", flush=True)
