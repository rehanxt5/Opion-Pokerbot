# PPO auto cell F: tether probe + beta (notebook cell 8)
import glob

import numpy as np

probe_t, probe_c, probe_p = [], [], []
need = 32768
for f in sorted(glob.glob("/content/packed/*/*/acts.npz")):
    z = np.load(f, allow_pickle=False)
    s = np.flatnonzero(z["part"] == 0)
    take = min(len(s), need - sum(len(x) for x in probe_t))
    if take <= 0:
        break
    s = s[:take]
    probe_t.append(z["tokens"][s])
    c = z["ctx"][s].astype(np.float32)
    c[:, 14] = 0.0
    probe_c.append(c)
    probe_p.append(z["p_soft"][s])
PT = torch.from_numpy(np.concatenate(probe_t)[:need].astype(np.int64)).cuda()
PC = torch.from_numpy(np.concatenate(probe_c)[:need].astype(np.float32)).cuda()
PP = torch.from_numpy(np.concatenate(probe_p)[:need].astype(np.float32)).cuda()
print("probe:", len(PT), "states")
BETA = float(open("/content/rl_iter/beta.txt").read().strip())
print("beta =", BETA)
print("CELL_F_DONE", flush=True)
