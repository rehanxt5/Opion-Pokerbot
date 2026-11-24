# Colab gate cell — torch vs numpy port differential test (§9.4).
# Prereqs in the session: T4 runtime; np_policy.py, model.py, final_act.pt
# uploaded (colab CLI or drag-drop); packed.tar via gdown (auto below).
# Verdict line: GATE_PASS / GATE_FAIL, plus GATE_RESULT json.
import glob
import json
import os
import sys
import tarfile
import time

import numpy as np
import torch

t0 = time.time()
for need in ("np_policy.py", "model.py", "final_act.pt"):
    assert os.path.exists(need), f"upload {need} to the session first"

# 1. packed states (real data, never the test drawer: use TRAIN rows part==0)
if not glob.glob("packed/*/*/acts.npz"):
    if not os.path.exists("packed.tar"):
        import gdown
        gdown.download(id="1AgcO3IHlhFc4czzr1fU_EG39Bh7809TA", output="packed.tar")
    with tarfile.open("packed.tar") as tf:
        tf.extractall("/content")

sys.path.insert(0, "/content")
import np_policy  # noqa: E402

N_TARGET = 10528
toks_l, ctxs_l = [], []
for f in sorted(glob.glob("packed/*/*/acts.npz")):
    z = np.load(f, allow_pickle=False)
    sel = (z["part"] == 0) & (z["street"] == 1)  # train rows; flop-biased mix
    idx = np.flatnonzero(sel)[: N_TARGET - sum(len(t) for t in toks_l)]
    if len(idx):
        toks_l.append(z["tokens"][idx])
        ctxs_l.append(z["ctx"][idx])
    if sum(len(t) for t in toks_l) >= N_TARGET:
        break
tokens = np.concatenate(toks_l)[:N_TARGET].astype(np.int64)
ctx = np.concatenate(ctxs_l)[:N_TARGET].astype(np.float32)
print(f"gate states: {len(tokens):,}")

# 2. canonical weight export: torch state_dict -> policy_np.npz
sd_t = torch.load("final_act.pt", map_location="cpu")["model"]
sd_np = {k: v.detach().to(torch.float32).cpu().numpy() for k, v in sd_t.items()}
np_policy.save_npz(sd_np, "policy_np.npz")
print(f"exported policy_np.npz ({os.path.getsize('policy_np.npz')/1e6:.1f} MB)")

# 3. no-torch .pt parser must agree bitwise with torch's own load
sd_pt = {k: np.asarray(v) for k, v in np_policy.load_pt("final_act.pt")["model"].items()}
parse_dmax = max(float(np.max(np.abs(sd_pt[k].astype(np.float64)
                                    - sd_np[k].astype(np.float64))))
                 for k in sd_np)
print(f"parser-vs-torch weight max|diff| = {parse_dmax:.3e}")

# 4. math gate: torch fp32 forward vs numpy port on real states
from model import OpionPolicy  # uploaded with the training files; fallback below

m = OpionPolicy()
m.load_state_dict(sd_t)
m.to("cuda").eval()
qs = []
with torch.no_grad():
    for i in range(0, len(tokens), 8192):
        sl = slice(i, i + 8192)
        qs.append(m(torch.from_numpy(tokens[sl]).cuda(),
                    torch.from_numpy(ctx[sl]).cuda()).float().cpu().numpy())
ref = np.concatenate(qs)

pol = np_policy.NumpyPolicy(np_policy.load_npz("policy_np.npz"))
port = np.empty_like(ref)
for i in range(0, len(tokens), 2048):
    sl = slice(i, i + 2048)
    port[sl] = pol.forward(tokens[sl], ctx[sl])

d = np.abs(ref.astype(np.float64) - port.astype(np.float64))
res = {
    "n_states": int(len(tokens)),
    "max_abs_logit_diff": float(d.max()),
    "mean_abs_logit_diff": float(d.mean()),
    "argmax_agreement": float((ref.argmax(-1) == port.argmax(-1)).mean()),
    "parser_weight_max_diff": parse_dmax,
    "wall_s": round(time.time() - t0, 1),
    "gate": bool(d.max() < 1e-3 and parse_dmax == 0.0
                 and (ref.argmax(-1) == port.argmax(-1)).all()),
}
print(json.dumps(res, indent=1))
print("GATE_PASS" if res["gate"] else "GATE_FAIL")
