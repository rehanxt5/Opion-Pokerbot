#!/usr/bin/env python3
"""amt-convention diagnosis, ON THE SANDBOX (8K packed train states).

A: forward with training ctx (ctx[14]=true amt)  -> sanity, should match quiz
B: forward with ctx[14]=0 forced (old live bot)  -> expect call-collapse
C: per-bucket assembly (new live bot): amt=0 row scores buckets 0/1, each
   legal raise bucket scored at its own decoded amount -> expect recovery
"""
import sys
import time

sys.path.insert(0, "/home/user/opion/rl")
import numpy as np  # noqa: E402

import np_policy  # noqa: E402
import spec  # noqa: E402

pol = np_policy.NumpyPolicy(np_policy.load_pt(
    "/home/user/opion/rl/current/final_act.pt")["model"])

import glob

_npz = (glob.glob("/home/user/opion/packed/w1/sb00/acts.npz")
        or glob.glob("/home/user/opion/packed/acts.npz"))[0]
z = np.load(_npz)
N = 4000
toks = z["tokens"][:N].astype(np.int64)
ctx0 = z["ctx"][:N].astype(np.float32)
legal = z["legal"][:N]
act_idx = z["act_idx"][:N]
street = z["street"][:N]
hand_ids = z["hand"][:N].astype(np.int64)


def mix(lg):
    a = np.where(legal != 0, lg, -np.inf).argmax(-1)
    return np.bincount(a, minlength=6) / len(a)


t0 = time.time()
A = pol.forward(toks, ctx0)
print("A true-amt  mix", np.round(mix(A), 3).tolist(),
      "ag", round(float((A.argmax(-1) == act_idx).mean()), 4),
      f"({time.time()-t0:.0f}s)")

t0 = time.time()
cb = ctx0.copy()
cb[:, 14] = 0.0
B = pol.forward(toks, cb)
print("B zero-amt  mix", np.round(mix(B), 3).tolist(),
      "ag", round(float((B.argmax(-1) == act_idx).mean()), 4),
      f"({time.time()-t0:.0f}s)")

t0 = time.time()
st_names = ["preflop", "flop", "turn", "river"]
rt, rc_, meta = [], [], []
for i in range(N):
    if not (legal[i, 2] or legal[i, 3] or legal[i, 4] or legal[i, 5]):
        continue
    pot = max(int(round(ctx0[i, 4] * 200)), 1)
    tc = int(round(ctx0[i, 5] * 200))
    req = {"legal_actions": (["fold", "call", "raise"] if tc > 0
                             else ["check", "raise"]),
           "to_call": tc, "street": st_names[street[i]], "pot": pot,
           "opp_bet": int(round(ctx0[i, 7] * 200)),
           "min_raise": int(round(ctx0[i, 10] * 200)),
           "max_raise": int(round(ctx0[i, 11] * 200)),
           "raises_left": int(round(ctx0[i, 12] * 3)),
           "hand_id": int(hand_ids[i]), "my_stack": 200}
    for b in (2, 3, 4, 5):
        if not legal[i, b]:
            continue
        rep = spec.decode_bucket(b, req)
        amt_b = int(rep.get("amount") or 0) \
            if rep.get("action") == "raise" else 0
        c = ctx0[i].copy()
        c[14] = amt_b / pot
        rt.append(toks[i])
        rc_.append(c)
        meta.append((i, b))
cb0 = ctx0.copy()
cb0[:, 14] = 0.0
L0 = pol.forward(toks, cb0)
LR = np.zeros((len(rt), 6), dtype=np.float32)
CH = 4096
for k in range(0, len(rt), CH):
    LR[k:k + CH] = pol.forward(np.stack(rt[k:k + CH]),
                               np.stack(rc_[k:k + CH]))
per = np.full((N, 6), -1e9, dtype=np.float64)
per[:, 0] = L0[:, 0]
per[:, 1] = L0[:, 1]
for (i, b), lg in zip(meta, LR):
    per[i, b] = lg[b]
C_ag = float((np.where(legal != 0, per, -np.inf).argmax(-1) == act_idx).mean())
print("C per-bkt   mix", np.round(mix(per), 3).tolist(),
      "ag", round(C_ag, 4), f"({time.time()-t0:.0f}s)")
for s in range(4):
    m = street == s
    ag = float((np.where(legal[m] != 0, per[m], -np.inf).argmax(-1)
                == act_idx[m]).mean())
    print(f"   C street {s}: ag={ag:.4f} n={int(m.sum())}")
ok = C_ag > 0.90 and abs(mix(per)[3] - mix(A)[3]) < 0.05
print("AMT_EXPERIMENT_" + ("CONFIRMED_FIX" if ok else "INCONCLUSIVE"))
