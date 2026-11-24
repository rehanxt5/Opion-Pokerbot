#!/usr/bin/env python3
"""opion census: per-opponent headcount of every teacher decision (is_t=true)
in the raw obs logs. Runs on a sandbox; stdlib only; the result JSON stays on
the sandbox disk (nothing is downloaded).

python3 census.py --data-dir /home/user/opion/data \
    [--out /home/user/opion/census.json] [--workers 8]

Counts, per opponent (label taken from each record's own opp_bot field):
  act decisions by street (0=preflop 1=flop 2=turn 3=river)
  disc decisions by street (derived from board width; flop-only by game rule)
  teacher seat balance (seat 0 = seat A)
  matches, unique seeds, unique hands
Every match file is cross-checked to carry exactly one opponent label.

Speed: 8 worker processes (8 vCPU), one match file each; lines are filtered
with a substring scan ('"is_t":true' — the engine writes compact JSON) and
only the teacher's ~half of records gets json.loads'd.
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

DISC_ST = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}  # board width -> street


def open_any(p):
    if p.endswith(".gz"):
        return gzip.open(p, "rb")
    return open(p, "rb")


def process_match(path):
    """One match file per worker: stream lines, count the teacher's records."""
    act = Counter()    # (opp, street) -> n
    disc = Counter()   # (opp, street) -> n
    seat = Counter()   # (opp, seat) -> n
    opps = set()
    seeds = set()
    hands = set()
    n_lines = n_t = 0
    with open_any(path) as f:
        for line in f:
            n_lines += 1
            if b'"is_t":true' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("is_t"):
                continue
            n_t += 1
            opp = r.get("opp_bot") or "?"
            opps.add(opp)
            seeds.add(r.get("seed"))
            # hand ids restart per leg (400-hand match = 2 mirror legs of 200)
            hands.add((r.get("leg"), r.get("hand")))
            seat[(opp, r.get("seat"))] += 1
            t = r.get("t")
            if t == "act":
                st = r.get("street")
                act[(opp, st if isinstance(st, int) and 0 <= st <= 3 else "?")] += 1
            elif t == "disc":
                disc[(opp, DISC_ST.get(len(r.get("board") or []), "?"))] += 1
    return {"path": path, "n_lines": n_lines, "n_t": n_t, "act": act,
            "disc": disc, "seat": seat, "opps": list(opps),
            "seeds": list(seeds), "hands": len(hands)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/user/opion/data")
    ap.add_argument("--out", default="/home/user/opion/census.json")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()

    t0 = time.monotonic()
    # dedupe obs files, preferring .gz (run_shard gzips but keeps the original)
    by_base = {}
    for o in sorted(glob.glob(os.path.join(a.data_dir, "obs_r*_m*.jsonl*"))):
        base = os.path.basename(o).replace(".gz", "")
        if base not in by_base or o.endswith(".gz"):
            by_base[base] = o
    jobs = sorted(by_base.values())
    if not jobs:
        print("CENSUS_FAILED no obs files under " + a.data_dir, flush=True)
        sys.exit(2)
    dupes = len(glob.glob(os.path.join(a.data_dir, "obs_r*_m*.jsonl*"))) - len(jobs)
    print(f"[census] {len(jobs)} matches ({dupes} .gz dupes skipped), "
          f"{a.workers} workers", flush=True)

    with Pool(a.workers) as pool:
        results = pool.map(process_match, jobs, chunksize=1)

    # ---- merge in parent
    g_act, g_disc, g_seat = Counter(), Counter(), Counter()
    opp_matches, opp_hands = Counter(), Counter()
    opp_seeds = {}
    per_match = []
    anomalies = []
    tot_lines = tot_t = 0
    for r in results:
        tot_lines += r["n_lines"]
        tot_t += r["n_t"]
        base = os.path.basename(r["path"])
        g_act.update(r["act"])
        g_disc.update(r["disc"])
        g_seat.update(r["seat"])
        for opp in r["opps"]:
            opp_matches[opp] += 1
            opp_hands[opp] += r["hands"]
            opp_seeds.setdefault(opp, set()).update(
                s for s in r["seeds"] if s)
        if len(r["opps"]) != 1 or r["n_t"] == 0:
            anomalies.append({"match": base, "opps": r["opps"],
                              "teacher": r["n_t"]})
        per_match.append({
            "match": base,
            "opp": r["opps"][0] if len(r["opps"]) == 1 else None,
            "seed": r["seeds"][0] if len(r["seeds"]) == 1 else None,
            "teacher": r["n_t"], "hands": r["hands"]})
    per_match.sort(key=lambda m: m["match"])

    def opp_sum(table, opp):
        return sum(v for (o, _s), v in table.items() if o == opp)

    opponents = {}
    for opp in sorted(opp_matches):
        o = {
            "matches": opp_matches[opp],
            "seeds": sorted(opp_seeds.get(opp, [])),
            "hands": opp_hands[opp],
            "act": opp_sum(g_act, opp),
            "disc": opp_sum(g_disc, opp),
            "act_by_street": [g_act.get((opp, s), 0) for s in range(4)],
            "disc_by_street": [g_disc.get((opp, s), 0) for s in range(4)],
            "act_unknown_street": g_act.get((opp, "?"), 0),
            "disc_unknown_street": g_disc.get((opp, "?"), 0),
            "seat0": g_seat.get((opp, 0), 0),
            "seat1": g_seat.get((opp, 1), 0),
        }
        o["total"] = o["act"] + o["disc"]
        o["seat_other"] = opp_sum(g_seat, opp) - o["seat0"] - o["seat1"]
        opponents[opp] = o

    tot_act = sum(g_act.values())
    tot_disc = sum(g_disc.values())
    out = {
        "schema": "opion-census-v1",
        "data_dir": a.data_dir,
        "host_cpus": os.cpu_count(),
        "workers": a.workers,
        "wall_s": round(time.monotonic() - t0, 1),
        "totals": {
            "files": len(jobs),
            "gz_dupes_skipped": max(0, dupes),
            "lines": tot_lines,
            "teacher_records": tot_t,
            "teacher_act": tot_act,
            "teacher_disc": tot_disc,
            "unique_seeds": len({s for ss in opp_seeds.values() for s in ss}),
            "unique_hands": sum(opp_hands.values()),
            "anomalies": anomalies,
        },
        "opponents": opponents,
        "per_match": per_match,
    }
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    # ---- human summary
    print(f"[census] lines={tot_lines:,} teacher={tot_t:,} "
          f"(act={tot_act:,} disc={tot_disc:,}) wall={out['wall_s']}s",
          flush=True)
    for opp, o in opponents.items():
        pct = 100.0 * o["total"] / max(1, tot_t)
        print(f"[census] {opp:<14} total={o['total']:>9,} ({pct:5.2f}%) "
              f"act={o['act']:>9,} disc={o['disc']:>7,} "
              f"streets={o['act_by_street']} hands={o['hands']:,} "
              f"matches={o['matches']} seat0/1={o['seat0']:,}/{o['seat1']:,}",
              flush=True)
    if anomalies:
        print(f"[census] ANOMALIES: {anomalies[:5]} "
              f"(+{max(0, len(anomalies) - 5)} more)", flush=True)
    print("CENSUS_OK", flush=True)


if __name__ == "__main__":
    main()
