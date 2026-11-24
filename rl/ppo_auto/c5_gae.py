# PPO auto cell E: load rollouts -> GAE advantages (notebook cell 7)
import glob
import json

import numpy as np

fs = sorted(glob.glob("/content/rl_iter/rollouts/*.npz"))
assert fs, "no rollouts in tar"
cols = {k: [] for k in ("tokens", "ctx", "legal", "bucket", "logp",
                        "pid", "hand", "seq", "street", "reward")}
for f in fs:
    z = np.load(f)
    for k in cols:
        cols[k].append(z[k])
D = {k: np.concatenate(v) for k, v in cols.items()}
D["ctx"][:, 14] = 0.0
print(f"rollouts: {len(fs)} shards, {len(D['bucket']):,} decisions, "
      f"{len(set(zip(D['pid'].tolist(), D['hand'].tolist()))):,} chains")

order = np.lexsort((D["seq"], D["hand"], D["pid"]))
for k in D:
    D[k] = D[k][order]

pid, hnd = D["pid"], D["hand"]
same = (pid[1:] == pid[:-1]) & (hnd[1:] == hnd[:-1])
chain = np.concatenate([[0], np.cumsum(~same)])
n_chains = int(chain[-1]) + 1

with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    vals = []
    for i in range(0, len(D["bucket"]), 16384):
        sl = slice(i, i + 16384)
        _, v = model.forward_both(torch.from_numpy(D["tokens"][sl]).cuda(),
                                  torch.from_numpy(D["ctx"][sl]).cuda())
        vals.append(v.float())
    V = torch.cat(vals).cpu().numpy().astype(np.float64)

is_last = np.concatenate([same, [False]]) == False  # noqa: E712
R = np.where(is_last, D["reward"], 0.0)  # terminal row of each chain = hand net

adv = np.zeros_like(V)
lastgae = 0.0
for t in range(len(V) - 1, -1, -1):
    cont = t < len(V) - 1 and bool(same[t])
    nextv = V[t + 1] if cont else 0.0
    delta = R[t] + nextv - V[t]
    lastgae = delta + 0.95 * lastgae if cont else delta
    adv[t] = lastgae
ret = adv + V
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
print(f"GAE done: {n_chains} chains, adv mean {adv.mean():.3f} sd {adv.std():.3f}")
print(f"reward/hand overall mean {D['reward'][is_last].mean():+.4f}")
mix = np.bincount(D["bucket"], minlength=6) / len(D["bucket"])
print("behavior bucket mix:", np.round(mix, 3).tolist())
print("CELL_E_DONE", flush=True)
