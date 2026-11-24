#!/usr/bin/env python3
"""Freeze-time field battery for RL iterations: run the top RL candidates
(student_bot.py + OPION_ACT_NPZ) against FLEET7 through the engine, same
harness as rl_fleet_battery.py, so their fieldEV joins the A/B/gen86
ranking directly. 12 matches per pair, 8 sandboxes, seeds rl:c:*.
Default candidates: 14 (champion screen), 24 (2nd-best screen), and the
latest trained iter found in ckpts. Override via argv: iter numbers."""
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

OPPS = {
    "elitev6": "bot/elite_instr.py",
    "elite83": "opponents/elite83.py",
    "maniac": "opponents/styles/maniac.py",
    "station": "opponents/styles/station.py",
    "rock": "opponents/styles/rock.py",
    "random": "opponents/styles/randombot.py",
    "mixer": "opponents/styles/mixer.py",
}
N_PER_PAIR = 12

NET_RE = re.compile(r"^\s+([AB])\s+.*net\s+([+-]?\d+)", re.M)


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def candidates():
    ck = os.path.join(ROOT, "opion/rl/ckpts")
    iters = sorted(int(f[len("policy_np_iter"):-len(".npz")])
                   for f in os.listdir(ck)
                   if f.startswith("policy_np_iter"))
    if len(sys.argv) > 1:
        want = [int(a) for a in sys.argv[1:]]
    else:
        want = [14, max(i for i in iters if i in (24, 28, 29, 30, 31))
                if any(i in iters for i in (24, 28, 29, 30, 31)) else iters[-1]]
        want = sorted(set(want))
    for w in want:
        assert w in iters, f"policy_np_iter{w}.npz missing"
    return want


def build_jobs(cands):
    jobs = []
    i = 0
    for c in cands:
        for opp in OPPS:
            for _ in range(N_PER_PAIR):
                jobs.append((i, c, opp, f"rl:c:{i}"))
                i += 1
    return jobs


def one(k, sb_id, chunk, cands):
    tag = f"sb{k}"
    rec = {"tag": tag, "matches": len(chunk)}
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
        bdir = "/home/user/opion/rl/battery_iters"
        sb.commands.run(f"mkdir -p {bdir}", timeout=60)
        for c in cands:
            with open(os.path.join(ROOT, f"opion/rl/ckpts/"
                                           f"policy_np_iter{c}.npz"),
                      "rb") as f:
                sb.files.write(f"{bdir}/policy_np_iter{c}.npz", f.read())
        rows = []
        for (gi, c, opp, seed) in chunk:
            cmd = (f"cd /home/user/opion && "
                   f"OPION_ACT_NPZ={bdir}/policy_np_iter{c}.npz "
                   f"OPION_DISC_PT=/home/user/opion/rl/current/final_disc.pt "
                   f"python3 engine_log.py rl/student_bot.py {OPPS[opp]} "
                   f"--stage 1 --hands 400 --seed {seed} "
                   f"--name-a i{c} --name-b {opp}")
            r = sb.commands.run(cmd, timeout=900)
            if r.exit_code != 0:
                rows.append({"gi": gi, "c": c, "opp": opp, "rc": r.exit_code,
                             "net_x": None})
                continue
            nets = dict((m.group(1), int(m.group(2)))
                        for m in NET_RE.finditer(r.stdout))
            rows.append({"gi": gi, "c": c, "opp": opp, "rc": 0,
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
        print(f"[bat2] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"matches={len(rec.get('rows', []))} "
              f"paused={rec.get('paused')} ({ok}/{len(RESULTS)}) "
              f"{rec.get('error', '')[:120]}", flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    cands = candidates()
    jobs = build_jobs(cands)
    print(f"[bat2] candidates {cands}, {len(jobs)} matches total", flush=True)
    summ = json.load(open(os.path.join(ROOT,
                                       "opion/data/rl/iter_0002_summary.json")))
    sandbox_ids = [r["sandbox"] for r in summ["sandboxes"]]
    chunks = [[] for _ in range(NSB)]
    for j in jobs:
        chunks[j[0] % NSB].append(j)
    for c in chunks:
        c.sort()
    t0 = time.monotonic()
    sem = threading.Semaphore(CONC)

    def worker(k, sid, chunk):
        with sem:
            one(k, sid, chunk, cands)

    ths = []
    for k, sid in enumerate(sandbox_ids):
        th = threading.Thread(target=worker, args=(k, sid, chunks[k]),
                              daemon=True)
        th.start()
        ths.append(th)
        time.sleep(2)
    while any(th.is_alive() for th in ths):
        time.sleep(45)
        with LOCK:
            print(f"[bat2 monitor] t={(time.monotonic()-t0)/60:.1f}min "
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
            per.setdefault((row["c"], row["opp"]), []).append(
                row["net_x"] / 400.0 / 200.0)  # stacks/hand
    out = {"candidates": cands, "per_match": {f"sb{k}": RESULTS[k].get("rows", [])
                                              for k in sorted(RESULTS)}}
    with open(os.path.join(ROOT, "opion/data/rl/rl_iters_battery.json"),
              "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== RL-ITERS FIELD BATTERY (stacks/hand vs FLEET7) ===", flush=True)
    for c in cands:
        tot_v = tot_n = 0.0
        line = []
        for opp in OPPS:
            vals = per.get((c, opp))
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
        print(f"  iter-{c:<3} fieldEV {ev:+.5f}  |  " + "  ".join(line),
              flush=True)
    print(f"[bat2] bad matches: {bad}", flush=True)
    print("reference fieldEV: gen86 +0.0379 | B +0.0373 | A +0.0294 | "
          "champ(rl:i15 basis) +0.0244", flush=True)
    print("BAT2_DONE", flush=True)


if __name__ == "__main__":
    main()
