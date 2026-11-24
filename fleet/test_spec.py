#!/usr/bin/env python3
"""spec.py validation battery (runs inside the sandbox on real shard data).

T1 decode_bucket parity: teacher's soft rep (action, amount) must equal
   spec.decode_bucket(soft.act_idx, req_echo) on every teacher act record.
T2 encode/decode roundtrip: encode_act -> decode_features -> re-encode of
   the lossless fields must be identical (cards, board, opp_disc, hist).
T3 invariants: legal mask never all-zero; p sums to 1; partitions cover.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402


def open_any(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/user/opion/data"
    soft_files = sorted(glob.glob(os.path.join(data_dir, "logs", "*",
                                               "soft.jsonl")))
    obs_files = sorted(glob.glob(os.path.join(data_dir, "obs_r*_m*.jsonl*")))
    print(f"[test] {len(obs_files)} obs files, {len(soft_files)} soft files")

    # soft index for the first N matches (memory-bounded test)
    soft = {}
    for sf in soft_files[:4]:
        with open(sf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("t") == "soft":
                    soft[(r.get("pid"), r.get("hand"), r.get("seq"))] = r

    t1_n = t1_ok = t2_n = t2_ok = t3_bad_mask = t3_bad_p = 0
    part_seen = set()
    limit = 60000
    n = 0
    for of in obs_files:
        if n >= limit:
            break
        with open_any(of) as f:
            for line in f:
                if n >= limit:
                    break
                if '"is_t":true' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["t"] != "act" or not r.get("is_t"):
                    continue
                s = soft.get((r.get("bpid"), r.get("hand"), r.get("seq")))
                if s is None:
                    continue
                n += 1
                req = r["req_echo"]
                # ---- T1 (amount compared only for raises — check/call
                # reps carry no amount key)
                rep = s.get("rep") or {}
                dec = spec.decode_bucket(s["act_idx"], req)
                t1_n += 1
                if dec.get("action") == rep.get("action") and \
                        (rep.get("action") != "raise" or
                         int(dec.get("amount", -1)) ==
                         int(rep.get("amount", -1))):
                    t1_ok += 1
                # ---- T2 + T3
                fe = spec.encode_act(r, s)
                p32 = fe["p_soft"].astype(np.float32)
                if abs(p32.sum() - 1.0) > 0.02:
                    t3_bad_p += 1
                if fe["legal"].sum() == 0:
                    t3_bad_mask += 1
                part_seen.add(int(fe["part"]))
                back = spec.decode_features(fe["tokens"], fe["ctx"])
                if back["hole"] == list(fe["tokens"][0:3]) and \
                        back["board"] == list(fe["tokens"][3:8]) and \
                        (back["opp_disc"] or None) == \
                        (int(fe["tokens"][8]) or None):
                    # hist re-encode check
                    re_tok = np.zeros(spec.N_TOK_ACT, dtype=np.uint8)
                    re_tok[0:3] = back["hole"]
                    re_tok[3:8] = back["board"]
                    re_tok[8] = back["opp_disc"] or 0
                    for i, (st, actor, ai, _a) in enumerate(back["hist"]):
                        re_tok[9 + i] = (st * 2 + actor) * 4 + ai + 1
                    t2_n += 1
                    if np.array_equal(re_tok, fe["tokens"]):
                        t2_ok += 1

    print(f"T1 decode_bucket vs teacher rep: {t1_ok}/{t1_n} "
          f"({t1_ok / max(1, t1_n) * 100:.3f}%)")
    print(f"T2 encode/decode roundtrip:      {t2_ok}/{t2_n}")
    print(f"T3 legal-mask all-zero:          {t3_bad_mask}")
    print(f"T3 p-sum violations:             {t3_bad_p}")
    print(f"T3 partitions seen:              {sorted(part_seen)}")
    ok = (t1_ok == t1_n and t2_ok == t2_n and t3_bad_mask == 0
          and t3_bad_p == 0 and part_seen and part_seen <= {0, 1, 2})
    print("SPEC_TEST_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
