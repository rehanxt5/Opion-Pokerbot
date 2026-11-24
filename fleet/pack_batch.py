#!/usr/bin/env python3
"""Phase-2 packer: one sandbox shard -> packed training arrays.

python3 pack_batch.py --data-dir /home/user/opion/data \
    --out-dir /home/user/opion/packed [--workers 8]

- Discovers every match (obs_rX_mY.jsonl[.gz] + logs/rX_mY/soft.jsonl)
- 8 worker processes, one match each (8 vCPU -> 8 matches per round)
- Joins soft labels by (bpid, hand, seq); encodes via spec.py
- HARD validation: soft coverage >= 99.9%, action agreement == 100%,
  p sums to 1 -> exits nonzero on failure
- Saves acts.npz / disc.npz / manifest.json in --out-dir (stays in the
  sandbox; nothing is downloaded)
"""
import argparse
import glob
import gzip
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402

ACT_FIELDS = ("tokens", "ctx", "legal", "p_soft", "cfr", "act_idx", "part",
              "hand", "seq", "bpid", "street", "amt")
DISC_FIELDS = ("tokens", "ctx", "target", "part", "hand", "street")


def open_any(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def load_soft(path):
    """(bpid, hand, seq) -> soft record"""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("t") == "soft":
                out[(r.get("pid"), r.get("hand"), r.get("seq"))] = r
    return out


def p_sum_ok(p):
    s = float(np.sum(p.astype(np.float32)))
    return 0.98 <= s <= 1.02


def one_hot_ok(p):
    m = float(np.max(p.astype(np.float32)))
    return m >= 0.999


def process_match(job):
    obs_path, soft_path = job
    n_obs = n_act = n_disc = 0
    n_soft_miss = n_bad_p = n_disagree = 0
    cols = {k: [] for k in ACT_FIELDS}
    dcols = {k: [] for k in DISC_FIELDS}
    seed_of_first = None
    soft = load_soft(soft_path) if soft_path and os.path.exists(soft_path) else {}
    with open_any(obs_path) as f:
        for line in f:
            if '"is_t":true' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("is_t"):
                continue
            n_obs += 1
            if seed_of_first is None:
                seed_of_first = r.get("seed")
            if r["t"] == "act":
                s = soft.get((r.get("bpid"), r.get("hand"), r.get("seq")))
                if s is None:
                    n_soft_miss += 1
                    continue
                fe = spec.encode_act(r, s)
                if not p_sum_ok(fe["p_soft"]):
                    n_bad_p += 1
                    continue
                rep = s.get("rep") or {}
                agree = rep.get("action") == r["act"] and (
                    r["act"] != "raise" or
                    int(rep.get("amount") or -1) == int(r["amt"] or -1))
                if not agree:
                    n_disagree += 1
                    continue
                n_act += 1
                for k in ACT_FIELDS:
                    cols[k].append(fe[k])
            elif r["t"] == "disc":
                n_disc += 1
                fe = spec.encode_disc(r)
                for k in DISC_FIELDS:
                    dcols[k].append(fe[k])
    out = {"match": os.path.basename(obs_path),
           "n_obs": n_obs, "n_act": n_act, "n_disc": n_disc,
           "n_soft_miss": n_soft_miss, "n_bad_p": n_bad_p,
           "n_disagree": n_disagree, "seed": seed_of_first}
    for k in ACT_FIELDS:
        out[k] = cols[k]
    for k in DISC_FIELDS:
        out["d_" + k] = dcols[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t0 = time.monotonic()
    # dedupe: run_shard gzips but keeps the original — prefer .gz per match
    by_base = {}
    for o in sorted(glob.glob(os.path.join(a.data_dir, "obs_r*_m*.jsonl*"))):
        base = os.path.basename(o).replace(".gz", "")
        if base not in by_base or o.endswith(".gz"):
            by_base[base] = o
    jobs = []
    for base, o in sorted(by_base.items()):
        mdir = base[len("obs_"):-len(".jsonl")]
        soft = os.path.join(a.data_dir, "logs", mdir, "soft.jsonl")
        jobs.append((o, soft))
    print(f"[pack] {len(jobs)} matches, {a.workers} workers", flush=True)

    with Pool(a.workers) as pool:
        results = pool.map(process_match, jobs)

    # ---- aggregate + hard validation
    total_act = total_disc = 0
    miss = bad_p = disagree = 0
    streets = np.zeros(4, dtype=np.int64)
    parts = np.zeros(3, dtype=np.int64)
    seeds = {}
    for r in results:
        total_act += r["n_act"]
        total_disc += r["n_disc"]
        miss += r["n_soft_miss"]
        bad_p += r["n_bad_p"]
        disagree += r["n_disagree"]
        if r["seed"] and r["seed"] not in seeds:
            seeds[r["seed"]] = len(seeds)

    def stack(key, dtype):
        lists = [r[key] for r in results]
        if not any(lists):
            return np.zeros((0,) if dtype != np.uint8 else (0,), dtype=dtype)
        return np.array([x for lst in lists for x in lst], dtype=dtype)

    A = {k: stack(k, {"tokens": np.uint8, "ctx": np.float16,
                      "legal": np.uint8, "p_soft": np.float16,
                      "cfr": np.float16, "act_idx": np.uint8,
                      "part": np.uint8, "hand": np.uint32,
                      "seq": np.uint16, "bpid": np.uint32,
                      "street": np.uint8, "amt": np.uint16}[k])
         for k in ACT_FIELDS}
    D = {k: stack("d_" + k, {"tokens": np.uint8, "ctx": np.float16,
                             "target": np.uint8, "part": np.uint8,
                             "hand": np.uint32, "street": np.uint8}[k])
         for k in DISC_FIELDS}
    if total_act:
        for s in A["street"]:
            streets[s] += 1
        for p in A["part"]:
            parts[p] += 1

    cov = 1.0 - (miss / max(1, miss + total_act))
    agr = 1.0 - (disagree / max(1, disagree + total_act))
    hard_fail = []
    if cov < 0.999:
        hard_fail.append(f"soft_coverage={cov:.5f}")
    if agr < 0.999:
        hard_fail.append(f"action_agreement={agr:.5f}")
    if bad_p > 0:
        hard_fail.append(f"bad_p_sum={bad_p}")
    if total_act + total_disc == 0:
        hard_fail.append("no_teacher_records")

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, "acts.npz"),
                        seed_map=json.dumps(seeds), **A)
    np.savez_compressed(os.path.join(a.out_dir, "disc.npz"), **D)
    manifest = {
        "schema": spec.SCHEMA_VERSION,
        "data_dir": a.data_dir,
        "matches": len(jobs),
        "teacher_act": int(total_act),
        "teacher_disc": int(total_disc),
        "total": int(total_act + total_disc),
        "soft_coverage": round(cov, 6),
        "action_agreement": round(agr, 6),
        "bad_p_sum": bad_p,
        "streets": streets.tolist(),
        "partitions": parts.tolist(),
        "unique_seeds": len(seeds),
        "bytes_per_decision": round(sum(
            A[k].nbytes for k in ACT_FIELDS) / max(1, total_act), 1),
        "out_files": ["acts.npz", "disc.npz"],
        "wall_s": round(time.monotonic() - t0, 1),
        "hard_fail": hard_fail,
    }
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1), flush=True)
    if hard_fail:
        print("PACK_FAILED " + "; ".join(hard_fail), flush=True)
        sys.exit(2)
    print("PACK_OK", flush=True)


if __name__ == "__main__":
    main()
