#!/usr/bin/env python3
"""Build submissions/opion.py: inject base64 weight blobs into the template.
Actor = policy_np_iter14.npz (RL champion) as-is; disc = final_disc.pt
converted to npz via the no-torch loader."""
import base64
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "opion/rl"))
import numpy as np  # noqa: E402
import np_policy  # noqa: E402

TPL = os.path.join(ROOT, "submissions/opion_template.py")
OUT = os.path.join(ROOT, "submissions/opion.py")

# actor: champion iter-14 npz, re-stored as float16 (compute stays fp32 —
# NumpyPolicy casts to float32 on load; fp16 mantissa rel err <= 4.9e-4)
def to_fp16_npz(sd):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.ascontiguousarray(v, dtype=np.float16)
                                for k, v in sd.items()})
    return buf.getvalue()


act_sd = dict(np.load(os.path.join(ROOT, "opion/rl/ckpts/policy_np_iter14.npz"),
                      allow_pickle=False))
disc_sd = np_policy.load_pt(os.path.join(ROOT, "opion/train/final_disc.pt"))["model"]
assert not any(np.isnan(np.asarray(v, dtype=np.float32)).any()
               for v in list(act_sd.values()) + list(disc_sd.values()))

# fp16 drift gate: max|dlogit| fp32-vs-fp16 weights on plausible states
rng = np.random.default_rng(7)
toks = np.zeros((4096, 41), dtype=np.int64)
deck = np.arange(1, 53)
for i in range(4096):
    d = rng.permutation(deck)[:8]
    toks[i, :3] = d[:3]
    nb = int(rng.integers(1, 6))
    toks[i, 3:3 + nb] = d[3:3 + nb]
    nh = int(rng.integers(0, 9))
    toks[i, 9:9 + nh] = rng.integers(1, 33, nh)
ctx = rng.random((4096, 17)).astype(np.float32)
ctx[:, :4] = 0.0
ctx[np.arange(4096), rng.integers(0, 4, 4096)] = 1.0
pol32 = np_policy.NumpyPolicy(act_sd)
pol16 = np_policy.NumpyPolicy({k: v.astype(np.float16) for k, v in act_sd.items()})
d1 = np.abs(pol32.forward(toks, ctx).astype(np.float64)
            - pol16.forward(toks, ctx).astype(np.float64)).max()
dtoks = np.zeros((4096, 8), dtype=np.int64)
for i in range(4096):
    d = rng.permutation(deck)[:8]
    dtoks[i, :3] = d[:3]
    nb = int(rng.integers(1, 6))
    dtoks[i, 3:3 + nb] = d[3:3 + nb]
dctx = rng.random((4096, 16)).astype(np.float32)
dctx[:, :4] = 0.0
dctx[np.arange(4096), rng.integers(0, 4, 4096)] = 1.0
disc32 = np_policy.NumpyPolicy(disc_sd, n_tok=8, n_ctx=16, n_act=3,
                               map_tokens=False)
disc16 = np_policy.NumpyPolicy({k: v.astype(np.float16) for k, v in disc_sd.items()},
                               n_tok=8, n_ctx=16, n_act=3, map_tokens=False)
d2 = np.abs(disc32.forward(dtoks, dctx).astype(np.float64)
            - disc16.forward(dtoks, dctx).astype(np.float64)).max()
print(f"fp16 drift: act max|dlogit|={d1:.4f}  disc max|dlogit|={d2:.4f}")
assert d1 < 0.05 and d2 < 0.05, "fp16 drift gate FAILED"

act = to_fp16_npz(act_sd)
disc = to_fp16_npz(disc_sd)
print(f"actor npz(f16): {len(act):,} B, {len(act_sd)} keys")
print(f"disc  npz(f16): {len(disc):,} B, {len(disc_sd)} keys")

a64 = base64.b64encode(act).decode()
d64 = base64.b64encode(disc).decode()

tpl = open(TPL).read()
assert "__ACT_B64__" in tpl and "__DISC_B64__" in tpl
out = tpl.replace("__ACT_B64__", a64).replace("__DISC_B64__", d64)
open(OUT, "w").write(out)
print(f"opion.py: {os.path.getsize(OUT):,} B "
      f"({os.path.getsize(OUT)/1e6:.2f} MB, limit 5.0 MB)")
