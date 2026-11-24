#!/usr/bin/env python3
"""opion dataset QC gate (plan §4.7): audits obs/meta/soft streams and emits
qc_report.json. Exits 0 only if every hard check passes.

Checks: parse/gzip integrity, leakage invariants, duplicates, teacher tagging,
soft-label coverage + validity, stratification, seed partitions, counts.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

STREET_W = {0: 0, 1: 3, 2: 4, 3: 5}


def open_any(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rt", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--teacher", default="elitev6")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.data_dir
    rep = {"dir": d, "hard_pass": True, "checks": {}}

    def hard(name, ok, detail=None):
        rep["checks"][name] = {"pass": bool(ok)}
        if detail is not None:
            rep["checks"][name]["detail"] = detail
        if not ok:
            rep["hard_pass"] = False

    # ---------- parse streams ----------
    def dedupe(files):
        """When both .jsonl and .jsonl.gz exist (gzip keeps originals), count once."""
        stems = {}
        for f in files:
            base = f[:-3] if f.endswith(".gz") else f
            if f.endswith(".gz") or base not in stems:
                stems[base] = f
        return list(stems.values())

    obs_files = dedupe(sorted(glob.glob(os.path.join(d, "**", "obs_*.jsonl*"), recursive=True)))
    meta_files = dedupe(sorted(glob.glob(os.path.join(d, "**", "meta_*.jsonl*"), recursive=True)))
    soft_files = dedupe(sorted(glob.glob(os.path.join(d, "**", "soft*.jsonl*"), recursive=True)))
    rep["files"] = {"obs": len(obs_files), "meta": len(meta_files), "soft": len(soft_files)}
    parse_err = 0
    obs, meta, soft = [], [], []
    for f in obs_files + meta_files + soft_files:
        try:
            fh = open_any(f)
            for line in fh:
                try:
                    r = json.loads(line)
                    if "/obs_" in f.replace("\\", "/"):
                        obs.append(r)
                    elif "/meta_" in f.replace("\\", "/"):
                        meta.append(r)
                    else:
                        soft.append(r)
                except Exception:
                    parse_err += 1
            fh.close()
        except Exception as e:
            rep["checks"][f"gzip_open:{os.path.basename(f)}"] = {"pass": False, "err": repr(e)[:100]}
            rep["hard_pass"] = False
    hard("parse_errors_zero", parse_err == 0, {"errors": parse_err,
         "obs": len(obs), "meta": len(meta), "soft": len(soft)})

    acts = [r for r in obs if r.get("t") == "act"]
    discs = [r for r in obs if r.get("t") == "disc"]
    deals = [r for r in meta if r.get("t") == "deal"]
    ends = [r for r in meta if r.get("t") == "end"]

    # ---------- leakage invariants (on obs only) ----------
    leak = Counter()
    for r in acts:
        st = r.get("street")
        if st not in STREET_W or len(r.get("board", [])) != STREET_W[st]:
            leak["board_width"] += 1
        known = set(r.get("hole", [])) | set(r.get("board", []))
        extra = (set(r.get("hole", [])) | set(r.get("board", []))) - known  # noop guard
        od = r.get("opp_disc")
        if st in (0, 1) and od is not None:
            leak["opp_disc_early"] += 1
        if st in (2, 3) and od is None:
            leak["opp_disc_missing_late"] += 1
        re = r.get("req_echo", {})
        if re.get("to_call", 0) != max(0, re.get("opp_bet", 0) - re.get("my_bet", 0)):
            leak["to_call_math"] += 1
        if not re.get("legal_actions"):
            leak["no_legal"] += 1
        if r.get("act") not in ("fold", "check", "call", "raise"):
            leak["bad_action"] += 1
        if r.get("act") == "raise" and re.get("legal_actions") and "raise" in re["legal_actions"]:
            if not (re.get("min_raise", 0) <= (r.get("amt") or 0) <= re.get("max_raise", 10 ** 9)):
                leak["raise_out_of_range"] += 1
    for r in discs:
        if len(r.get("hole", [])) != 3:
            leak["disc_hole_not3"] += 1
        if r.get("card") not in r.get("hole", []):
            leak["disc_card_not_in_hole"] += 1
        if "opp_disc" in r:
            leak["disc_has_opp_field"] += 1
    hard("leakage_invariants", sum(leak.values()) == 0, dict(leak))

    # ---------- duplicates ----------
    act_k = Counter((r.get("seed"), r.get("leg"), r.get("hand"), r.get("seat"), r.get("seq"))
                    for r in acts)
    disc_k = Counter((r.get("seed"), r.get("leg"), r.get("hand"), r.get("seat"))
                     for r in discs)
    deal_k = Counter((r.get("seed"), r.get("leg"), r.get("hand")) for r in deals)
    dup = {"act": sum(c - 1 for c in act_k.values() if c > 1),
           "disc": sum(c - 1 for c in disc_k.values() if c > 1),
           "deal": sum(c - 1 for c in deal_k.values() if c > 1)}
    hard("no_duplicates", sum(dup.values()) == 0, dup)

    # ---------- teacher tagging & counts ----------
    t_acts = [r for r in acts if r.get("is_t")]
    t_discs = [r for r in discs if r.get("is_t")]
    rep["teacher_decisions"] = {"act": len(t_acts), "disc": len(t_discs),
                                "total": len(t_acts) + len(t_discs)}
    hard("teacher_tagged", len(t_acts) > 0 and len(t_discs) > 0,
         {"by_street": Counter(r["street"] for r in t_acts),
          "by_opponent": Counter(r.get("opp_bot", "?") for r in t_acts + t_discs),
          "by_seat": Counter(r.get("seat") for r in t_acts)})

    # seat balance (teacher as A vs B should be ~50/50 via alternating pairing)
    seats = Counter(r.get("seat") for r in t_acts)
    if sum(seats.values()) > 500:
        lo = min(seats.get(0, 0), seats.get(1, 0)) / max(1, sum(seats.values()))
        hard("seat_balance", lo >= 0.35, dict(seats))

    # ---------- soft-label coverage & validity ----------
    by_key = {}
    for r in soft:
        by_key[(r.get("pid"), r.get("hand"), r.get("seq"))] = r
    joined = sum(1 for r in t_acts if (r.get("bpid"), r.get("hand"), r.get("seq")) in by_key)
    cov = joined / max(1, len(t_acts))
    hard("soft_coverage", cov >= 0.99, {"joined": joined, "teacher_acts": len(t_acts),
                                        "coverage": round(cov, 4)})
    bad_p = 0
    agree = 0
    mode_n = Counter()
    for r in t_acts:
        s = by_key.get((r.get("bpid"), r.get("hand"), r.get("seq")))
        if not s:
            continue
        mode_n[s.get("mode")] += 1
        p = s.get("p", {})
        if s.get("mode") == "sample" and abs(sum(p.values()) - 1.0) > 1e-2:
            bad_p += 1
        if s.get("mode") in ("argmax", "rule") and list(p.values()) != [1.0]:
            bad_p += 1
        if s.get("rep", {}).get("action") == r.get("act") and \
           s.get("rep", {}).get("amount") == r.get("amt"):
            agree += 1
    hard("soft_probability_valid", bad_p == 0, {"violations": bad_p, "modes": dict(mode_n)})
    hard("soft_action_agreement", agree == len(t_acts),
         {"agree": agree, "total": len(t_acts)})

    # ---------- stratification ----------
    def pot_bucket(p):
        return "small" if p < 10 else ("mid" if p < 60 else "big")
    faces = Counter("facing_bet" if r.get("req_echo", {}).get("to_call", 0) > 0 else "checked_to"
                    for r in t_acts)
    threeb = Counter("3bet" if sum(1 for h in r.get("hist", [])
                                   if h[0] == 0 and h[2] == 3) >= 2 else "other"
                     for r in t_acts)
    pots = Counter(pot_bucket(r.get("req_echo", {}).get("pot", 0)) for r in t_acts)
    rep["stratification"] = {"facing": dict(faces), "pot": dict(pots),
                             "preflop_raises": dict(threeb),
                             "streets": dict(Counter(r["street"] for r in t_acts)),
                             "opponents": dict(Counter(r.get("opp_bot", "?") for r in t_acts))}

    # ---------- seed partitions (train/val/test at match-seed level) ----------
    def split_of(seed):
        h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16) % 100
        return "train" if h < 92 else ("val" if h < 96 else "test")
    part = Counter()
    part_type = defaultdict(Counter)
    for r in t_acts:
        sp = split_of(r.get("seed"))
        part[sp] += 1
        part_type[sp][r.get("opp_bot", "?")] += 1
    rep["partitions"] = {"counts": dict(part), "by_type": {k: dict(v) for k, v in part_type.items()}}
    hard("partitions_cover_all_seeds",
         sum(part.values()) == len(t_acts) and len(part) > 0)

    # ---------- hand-consistency: obs ↔ meta counts ----------
    hands_obs = {(r.get("seed"), r.get("leg"), r.get("hand")) for r in acts}
    hands_meta = {(r.get("seed"), r.get("leg"), r.get("hand")) for r in deals}
    hard("hands_match_meta", hands_obs <= hands_meta,
         {"obs_hands": len(hands_obs), "meta_deals": len(hands_meta)})

    rep["overall"] = "PASS" if rep["hard_pass"] else "FAIL"
    out = args.out or os.path.join(d, "qc_report.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print("QC", rep["overall"], "| teacher_decisions:", rep["teacher_decisions"]["total"],
          "| report:", out)
    if rep["hard_pass"] is False:
        for k, v in rep["checks"].items():
            if not v.get("pass"):
                print("  FAIL:", k, v.get("detail", ""))
    sys.exit(0 if rep["hard_pass"] else 1)


if __name__ == "__main__":
    main()
