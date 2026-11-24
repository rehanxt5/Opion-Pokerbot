#!/usr/bin/env python3
"""Merge the 40 per-sandbox census summaries into the fleet-wide table.

Reads opion/data/census/w{1,2}_sb*_census.json (trimmed summaries; the full
census.json files stay on the sandboxes) and writes census_fleet.json with
per-opponent decisions/act/disc/streets/seats/hands across the whole
25.4M-decision dataset.
"""
import glob
import json
import os

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUTDIR = os.path.join(ROOT, "opion", "data", "census")

KEYS = ("total", "act", "disc", "hands", "matches", "seat0", "seat1")
STREETS = ("act_by_street", "disc_by_street")


def main():
    files = sorted(glob.glob(os.path.join(OUTDIR, "w?_sb*_census.json")))
    assert len(files) == 40, f"expected 40 census summaries, got {len(files)}"
    agg = {}
    tot = {"decisions": 0, "act": 0, "disc": 0, "hands": 0, "matches": 0,
           "seeds": set()}
    for f in files:
        o = json.load(open(f))
        tot["decisions"] += o["totals"]["teacher_records"]
        tot["act"] += o["totals"]["teacher_act"]
        tot["disc"] += o["totals"]["teacher_disc"]
        tot["seeds"].update(s for d in o["opponents"].values()
                            for s in d.get("seeds", []))
        for name, d in o["opponents"].items():
            a = agg.setdefault(name, {k: 0 for k in KEYS}
                               | {s: [0, 0, 0, 0] for s in STREETS})
            for k in KEYS:
                a[k] += d.get(k, 0)
            for s in STREETS:
                for i in range(4):
                    a[s][i] += d.get(s, [0, 0, 0, 0])[i]

    opponents = {}
    for name, a in agg.items():
        opponents[name] = dict(a)
        opponents[name]["share_pct"] = round(100.0 * a["total"]
                                             / tot["decisions"], 3)
    out = {
        "schema": "opion-census-fleet-v1",
        "sandboxes": len(files),
        "totals": {"decisions": tot["decisions"], "act": tot["act"],
                   "disc": tot["disc"],
                   "hands": sum(a["hands"] for a in agg.values()),
                   "unique_seeds": len(tot["seeds"]),
                   "unique_opponents": len(agg)},
        "opponents": dict(sorted(opponents.items(),
                                 key=lambda kv: -kv[1]["total"])),
    }
    with open(os.path.join(OUTDIR, "census_fleet.json"), "w") as f:
        json.dump(out, f, indent=1)

    t = tot["decisions"]
    print(f"opion opponent census — {len(files)} sandboxes, "
          f"{t:,} teacher decisions, {len(agg)} opponents\n")
    print(f"{'opponent':<16}{'decisions':>11}{'share':>8}"
          f"{'act':>11}{'disc':>9}   preflop-flop-turn-river")
    for name, d in out["opponents"].items():
        st = "/".join(f"{v / 1000:.0f}k" if v >= 1000 else str(v)
                      for v in d["act_by_street"])
        print(f"{name:<16}{d['total']:>11,}{d['share_pct']:>7.2f}%"
              f"{d['act']:>11,}{d['disc']:>9,}   {st}")
    print(f"\nact={tot['act']:,} disc={tot['disc']:,} "
          f"hands={sum(a['hands'] for a in agg.values()):,} "
          f"unique_seeds={len(tot['seeds'])}")
    print("CENSUS_FLEET_DONE")


if __name__ == "__main__":
    main()
