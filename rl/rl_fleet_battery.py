#!/usr/bin/env python3
"""Field battery: score A, B and gen86 against the 7 fleet opponents from
the §9.3 screen mix (elitev6, elite83, maniac, station, rock, random,
mixer — gen86 skips the gen86 mirror). 12 matches per pair, 8 sandboxes,
fresh seeds rl:f:*. Champion (iter-14) numbers come from the rl:i15
screen for the same opponents. Output: per-candidate field EV ranking."""
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONC = 8
NSB = 8
RESULTS = {}
LOCK = threading.Lock()

BOTS = {
    "elitev6": "bot/elite_instr.py",
    "gen86": "opponents/gen86_pkg/package_g18_6/bot.py",
    "elite83": "opponents/elite83.py",
    "maniac": "opponents/styles/maniac.py",
    "station": "opponents/styles/station.py",
    "rock": "opponents/styles/rock.py",
    "random": "opponents/styles/randombot.py",
    "mixer": "opponents/styles/mixer.py",
    "A": "opponents/ext/sub_a/bot.py",
    "B": "opponents/ext/sub_b/bot.py",
}
FLEET7 = ["elitev6", "elite83", "maniac", "station", "rock", "random",
          "mixer"]
CANDIDATES = {"A": FLEET7, "B": FLEET7,
              "gen86": [o for o in FLEET7 if o != "gen86"]}
N_PER_PAIR = 12

B_FILES = ("bot.py", "mcbase.py", "pbcore.py", "synergy.py", "delaa_dump.py",
           "tns.py", "synergy_probe.py", "nnu.py", "tmr.py", "exbr.py")
B_DATA = ("tns_w.npz", "nnu_w.npz", "preflop_equity.npz")

NET_RE = re.compile(r"^\s+([AB])\s+.*net\s+([+-]?\d+)", re.M)


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def build_jobs():
    jobs = []
    i = 0
    for cand, opps in CANDIDATES.items():
        for opp in opps:
            for _ in range(N_PER_PAIR):
                jobs.append((i, cand, opp, f"rl:f:{i}"))
                i += 1
    return jobs


def one(k, sb_id, chunk):
    tag = f"sb{k}"
    rec = {"tag": tag, "sandbox": sb_id, "matches": len(chunk)}
    from e2b import Sandbox
    sb = None
    try:
        last = None
        for attempt in range(4):
            try:
                sb = Sandbox.connect(sb_id, timeout=300)
                break
            except Exception as e:
                last = e
                time.sleep(30 * (attempt + 1))
        if sb is None:
            raise RuntimeError(f"connect failed: {last!r}")
        sb.set_timeout(5400)
        op = "/home/user/opion/opponents"
        sb.commands.run("mkdir -p /home/user/opion/opponents/ext/sub_a "
                        "/home/user/opion/opponents/ext/sub_b/data", timeout=60)
        with open("/tmp/opp_inspect/a/bot.py", "rb") as f:
            sb.files.write(f"{op}/ext/sub_a/bot.py", f.read())
        for fn in B_FILES:
            with open(f"/tmp/opp_inspect/b/{fn}", "rb") as f:
                sb.files.write(f"{op}/ext/sub_b/{fn}", f.read())
        for fn in B_DATA:
            with open(f"/tmp/opp_inspect/b/data/{fn}", "rb") as f:
                sb.files.write(f"{op}/ext/sub_b/data/{fn}", f.read())
        rows = []
        for (gi, x, y, seed) in chunk:
            cmd = (f"cd /home/user/opion && python3 engine_log.py "
                   f"{BOTS[x]} {BOTS[y]} --stage 1 --hands 400 --seed {seed} "
                   f"--name-a {x} --name-b {y}")
            r = sb.commands.run(cmd, timeout=900)
            if r.exit_code != 0:
                rows.append({"gi": gi, "x": x, "y": y, "rc": r.exit_code,
                             "net_x": None})
                continue
            nets = dict((m.group(1), int(m.group(2)))
                        for m in NET_RE.finditer(r.stdout))
            rows.append({"gi": gi, "x": x, "y": y, "rc": 0,
                         "net_x": nets.get("A")})
        rec["rows"] = rows
        rec["ok"] = True
    except Exception as e:
        rec["error"] = repr(e)[:300]
        rec["ok"] = False
    if sb is not None:
        for attempt in range(4):
            try:
                sb.beta_pause()
                rec["paused"] = True
                break
            except Exception:
                time.sleep(20 * (attempt + 1))
        else:
            rec["paused"] = False
    with LOCK:
        RESULTS[k] = rec
        ok = sum(1 for x in RESULTS.values() if x.get("ok"))
        print(f"[bat] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"matches={len(rec.get('rows', []))} "
              f"paused={rec.get('paused')} ({ok}/{len(RESULTS)} done) "
              f"{rec.get('error', '')[:120]}", flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    summ = json.load(open(os.path.join(ROOT,
                                       "opion/data/rl/iter_0002_summary.json")))
    sandbox_ids = [r["sandbox"] for r in summ["sandboxes"]]
    assert len(sandbox_ids) == NSB
    jobs = build_jobs()
    print(f"[bat] {len(jobs)} matches total", flush=True)
    chunks = [[] for _ in range(NSB)]
    for j in jobs:
        chunks[j[0] % NSB].append(j)
    for c in chunks:
        c.sort()
    t0 = time.monotonic()
    sem = threading.Semaphore(CONC)

    def worker(k, sid, chunk):
        with sem:
            one(k, sid, chunk)

    ths = []
    for k, sid in enumerate(sandbox_ids):
        th = threading.Thread(target=worker, args=(k, sid, chunks[k]),
                              daemon=True)
        th.start()
        ths.append(th)
        time.sleep(2)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            print(f"[bat monitor] t={(time.monotonic()-t0)/60:.1f}min "
                  f"done={len(RESULTS)}/8", flush=True)
    for th in ths:
        th.join()

    per = {}
    bad = 0
    for k in sorted(RESULTS):
        for row in RESULTS[k].get("rows", []):
            if row["rc"] != 0 or row.get("net_x") is None:
                bad += 1
                continue
            per.setdefault((row["x"], row["y"]), []).append(
                row["net_x"] / 400.0 / 200.0)  # stacks/hand
    out = {"per_match": {f"sb{k}": RESULTS[k].get("rows", [])
                         for k in sorted(RESULTS)}}
    with open(os.path.join(ROOT, "opion/data/rl/fleet_battery.json"),
              "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== FIELD BATTERY (candidate stacks/hand vs each fleet opp, "
          "mirror legs) ===", flush=True)
    field_ev = {}
    for cand, opps in CANDIDATES.items():
        tot_v = tot_n = 0.0
        line = []
        for opp in opps:
            vals = per.get((cand, opp))
            if not vals:
                line.append(f"{opp}: n/a")
                continue
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            se = (var / len(vals)) ** 0.5
            n = len(vals) * 400
            tot_v += m * n
            tot_n += n
            line.append(f"{opp} {m:+.4f}±{se:.4f}")
        ev = tot_v / tot_n if tot_n else float("nan")
        field_ev[cand] = ev
        print(f"  {cand:<6} fieldEV {ev:+.5f}  |  " + "  ".join(line),
              flush=True)
    print(f"[bat] bad matches: {bad}", flush=True)
    print("Champion iter-14 reference (rl:i15 screen, same opps): "
          "elitev6 -0.0025 gen86 +0.0018 elite83 -0.0037 maniac +0.0683 "
          "station +0.0458 rock +0.0048 random +0.0471 mixer +0.0395",
          flush=True)
    champ = [(-0.0025 * 16000 + 0.0018 * 12000 - 0.0037 * 12000 +
              0.0683 * 8000 + 0.0458 * 8000 + 0.0048 * 8000 +
              0.0471 * 8000 + 0.0395 * 8000) / 76000]
    print(f"  champ  fieldEV {champ[0]:+.5f}  (hands-weighted, rl:i15)",
          flush=True)
    print("BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
