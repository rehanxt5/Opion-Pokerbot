# PPO auto cell D: torch vs numpy diff gate (notebook cell 6)
import glob
import sys

sys.path.insert(0, "/content/rl_iter")
import numpy as np
import np_policy

z0 = np.load(sorted(glob.glob("/content/packed/*/*/acts.npz"))[0],
             allow_pickle=False)
sel = np.flatnonzero(z0["part"] == 0)[:2048]
tk = z0["tokens"][sel].astype(np.int64)
cx = z0["ctx"][sel].astype(np.float32)
cx[:, 14] = 0.0
with torch.no_grad():
    ref = model.forward(torch.from_numpy(tk).cuda(),
                        torch.from_numpy(cx).cuda()).cpu().numpy()
pol = np_policy.NumpyPolicy({k: v.detach().cpu().numpy()
                             for k, v in model.state_dict().items()
                             if not k.startswith("value.")})
port = np.concatenate([pol.forward(tk[i:i + 1024], cx[i:i + 1024])
                       for i in range(0, len(tk), 1024)])
dmax = float(np.abs(ref.astype(np.float64) - port.astype(np.float64)).max())
print(f"diff gate: max|dlogit|={dmax:.2e}")
assert dmax < 1e-3, "numpy port gate FAILED"
print("GATE_PASS")
print("CELL_D_DONE", flush=True)
