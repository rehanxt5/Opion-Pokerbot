#!/usr/bin/env python3
"""Local test battery for the numpy port (no torch needed).

T0  erf approximation accuracy vs math.erf
T1  load real final_act.pt WITHOUT torch: param count + key/shape audit
T2  independent slow reference (loops + math.erf) vs vectorized port
T3  real packed states: forward invariants + masked-argmax sanity
T4  latency: single-decision and batch-64 forward (1 thread)
T5  _map_tokens edge cases

Run: python3 test_np_policy.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from np_policy import (D, FFN, HEADS, LAYERS, N_ACT, N_CTX, N_TOK,  # noqa: E402
                       DH, NumpyPolicy, _gelu, _layer_norm, _map_tokens,
                       _np_erf, load_pt)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PT = os.path.join(ROOT, "opion", "train", "final_act.pt")
NPZ_GLOB = os.path.join(ROOT, "opion", "data", "packed", "w1", "sb00", "acts.npz")

EXPECT_PARAMS = 812584
EXPECT_KEYS = (
    ["emb.weight", "pos.weight", "ctx_proj.0.weight", "ctx_proj.0.bias",
     "ctx_proj.1.weight", "ctx_proj.1.bias", "ln_f.weight", "ln_f.bias",
     "head.weight", "head.bias"]
    + [f"blocks.{i}.{m}.{t}" for i in range(4)
       for m in ("ln1", "qkv", "proj", "ln2", "fc1", "fc2")
       for t in ("weight", "bias")])


def slow_forward(sd, tokens, ctx):
    """Independent 1-sample reference: explicit loops, exact math.erf.
    Written differently on purpose — catches vectorization bugs."""
    def ln(v, w, b):
        m = sum(v) / len(v)
        var = sum((y - m) ** 2 for y in v) / len(v)
        return [(y - m) / math.sqrt(var + 1e-5) * w[j] + b[j]
                for j, y in enumerate(v)]
    tok = list(_map_tokens(tokens))
    x = [list(sd["emb.weight"][tok[i]] + sd["pos.weight"][i])
         for i in range(N_TOK)]
    cx = ln([float(c) for c in ctx], sd["ctx_proj.0.weight"],
            sd["ctx_proj.0.bias"])
    cx = [sum(cx[j] * sd["ctx_proj.1.weight"][o][j]
              for j in range(N_CTX)) + sd["ctx_proj.1.bias"][o]
          for o in range(D)]
    for i in range(N_TOK):
        for d in range(D):
            x[i][d] += cx[d]
    for L in range(LAYERS):
        p = f"blocks.{L}."
        # attention
        h = [ln(x[i], sd[p + "ln1.weight"], sd[p + "ln1.bias"])
             for i in range(N_TOK)]
        qkv = [[sum(h[i][d] * sd[p + "qkv.weight"][o][d]
                    for d in range(D)) + sd[p + "qkv.bias"][o]
                for o in range(3 * D)] for i in range(N_TOK)]
        att_out = [[0.0] * D for _ in range(N_TOK)]
        for hh in range(HEADS):
            q = [[qkv[i][hh * DH + d] for d in range(DH)] for i in range(N_TOK)]
            k = [[qkv[i][D + hh * DH + d] for d in range(DH)]
                 for i in range(N_TOK)]
            v = [[qkv[i][2 * D + hh * DH + d] for d in range(DH)]
                 for i in range(N_TOK)]
            for a in range(N_TOK):
                sc = [sum(q[a][d] * k[b][d] for d in range(DH)) / math.sqrt(DH)
                      for b in range(N_TOK)]
                mx = max(sc)
                ex = [math.exp(s - mx) for s in sc]
                tot = sum(ex)
                for b in range(N_TOK):
                    wgt = ex[b] / tot
                    for d in range(DH):
                        att_out[a][hh * DH + d] += wgt * v[b][d]
        for i in range(N_TOK):
            pr = [sum(att_out[i][d] * sd[p + "proj.weight"][o][d]
                      for d in range(D)) + sd[p + "proj.bias"][o]
                  for o in range(D)]
            for d in range(D):
                x[i][d] += pr[d]
        # ffn
        h = [ln(x[i], sd[p + "ln2.weight"], sd[p + "ln2.bias"])
             for i in range(N_TOK)]
        for i in range(N_TOK):
            f1 = [0.0] * FFN
            for o in range(FFN):
                s = sum(h[i][d] * sd[p + "fc1.weight"][o][d]
                        for d in range(D)) + sd[p + "fc1.bias"][o]
                f1[o] = 0.5 * s * (1.0 + math.erf(s / math.sqrt(2.0)))
            pr = [sum(f1[d] * sd[p + "fc2.weight"][o][d]
                      for d in range(FFN)) + sd[p + "fc2.bias"][o]
                  for o in range(D)]
            for d in range(D):
                x[i][d] += pr[d]
    xf = [ln(x[i], sd["ln_f.weight"], sd["ln_f.bias"]) for i in range(N_TOK)]
    pooled = [sum(xf[i][d] for i in range(N_TOK)) / N_TOK for d in range(D)]
    return [sum(pooled[d] * sd["head.weight"][o][d] for d in range(D))
            + sd["head.bias"][o] for o in range(N_ACT)]


def main():
    t0 = time.time()
    # T0: erf accuracy
    xs = np.linspace(-6, 6, 200_001)
    err = float(np.max(np.abs(_np_erf(xs)
                              - np.vectorize(math.erf)(xs))))
    print(f"T0 erf max|err|={err:.2e}  ({'PASS' if err < 2e-7 else 'FAIL'})")
    assert err < 2e-7

    # T1: real weights, no torch
    ck = load_pt(PT)
    sd = {k: np.asarray(v) for k, v in dict(ck["model"]).items()}
    n_params = int(sum(v.size for v in sd.values()))
    missing = [k for k in EXPECT_KEYS if k not in sd]
    print(f"T1 params={n_params:,} (expect {EXPECT_PARAMS:,}) "
          f"keys={len(sd)} missing={missing} "
          f"({ 'PASS' if n_params == EXPECT_PARAMS and not missing else 'FAIL'})")
    assert n_params == EXPECT_PARAMS and not missing
    assert sd["emb.weight"].shape == (85, D)
    assert sd["head.weight"].shape == (N_ACT, D)
    assert sd["blocks.0.qkv.weight"].shape == (3 * D, D)
    assert sd["blocks.0.fc1.weight"].shape == (FFN, D)

    pol = NumpyPolicy(sd)

    # T2: slow reference vs fast port on 3 real states
    z = np.load(NPZ_GLOB, allow_pickle=False)
    idx = np.array([0, len(z["tokens"]) // 2, len(z["tokens"]) - 1])
    toks = z["tokens"][idx].astype(np.int64)
    ctxs = z["ctx"][idx].astype(np.float32)
    fast = pol.forward(toks, ctxs)
    dmax = 0.0
    for b in range(3):
        ref = np.array(slow_forward(sd, toks[b], ctxs[b]), dtype=np.float64)
        dmax = max(dmax, float(np.max(np.abs(ref - fast[b].astype(np.float64)))))
    print(f"T2 slow-ref vs port max|dlogit|={dmax:.2e} "
          f"({'PASS' if dmax < 1e-5 else 'FAIL'})")
    assert dmax < 1e-5

    # T3: invariants on 512 real states
    n = 512
    toks = z["tokens"][:n].astype(np.int64)
    ctxs = z["ctx"][:n].astype(np.float32)
    legal = z["legal"][:n]
    logits = pol.forward(toks, ctxs)
    assert np.isfinite(logits).all()
    probs = pol.act_probs(toks, ctxs, legal)
    sums = probs.sum(axis=1)
    pick = probs.argmax(axis=1)
    legal_pick = legal[np.arange(n), pick].astype(bool)
    am = np.bincount(pick, minlength=6) / n
    print(f"T3 logits finite, prob-rows sum {sums.min():.6f}..{sums.max():.6f}, "
          f"argmax on-legal {legal_pick.mean() * 100:.1f}%")
    print(f"   masked-argmax mix {np.round(am, 3).tolist()}")
    assert abs(sums.max() - 1) < 1e-9 and abs(sums.min() - 1) < 1e-9
    assert legal_pick.all(), "masked argmax picked an illegal action"

    # T4: latency (1 thread)
    t = time.time()
    for i in range(100):
        pol.forward(toks[i % n:i % n + 1], ctxs[i % n:i % n + 1])
    single = (time.time() - t) / 100 * 1000
    t = time.time()
    for _ in range(20):
        pol.forward(toks, ctxs)
    b64 = (time.time() - t) / 20 / n * 1000
    print(f"T4 latency: single={single:.1f} ms/decision, "
          f"batch512={b64:.2f} ms/decision")

    # T5: token mapping edges
    # T5: token mapping — model.py rule: t<53 unchanged; else 52+clamp(t,1,32).
    # Raw data domain is cards 0..52 + hist tokens 1..32, all < 53, so the
    # effective map is IDENTITY there (embedding rows 1..32 are shared between
    # card ids and hist token ids — disambiguated by slot position embeddings;
    # quirk documented in plan.md §9.4, consistent train/eval/port).
    t5 = np.array([[0, 1, 52, 53, 84, 32, 1, 0, 0, 0] + [0] * 31], dtype=np.int64)
    m = _map_tokens(t5)[0][:7].tolist()
    exp = [0, 1, 52, 84, 84, 32, 1]
    print(f"T5 map_tokens {m} == {exp} ({'PASS' if m == exp else 'FAIL'})")
    assert m == exp

    print(f"ALL_LOCAL_TESTS_PASS in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
