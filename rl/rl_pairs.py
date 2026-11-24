#!/usr/bin/env python3
"""Round-robin legs for the external submissions: gen86 vs A, gen86 vs B,
A vs B (24 matches each, 8 sandboxes, fresh seeds rl:p:*). Engine run
directly per match (no student involved); per-match nets parsed from the
engine's stdout summary. Sandboxes re-paused after."""
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
    "gen86": "opponents/gen86_pkg/package_g18_6/bot.py",
    "A": "opponents/ext/sub_a/bot.py",
    "B": "opponents/ext/sub_b/bot.py",
}
PAIRS = [("gen86", "A"), ("gen86", "B"), ("A", "B")]
N_PER_PAIR = 24

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
    for (x, y) in PAIRS:
        for _ in range(N_PER_PAIR):
            jobs.append((i, x, y, f"rl:p:{i}"))
            i += 1
    assert i == 72
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
                         "net_x": nets.get("A"), "net_y": nets.get("B")})
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
        n = len(rec.get("rows", []))
        print(f"[pairs] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"matches={n} paused={rec.get('paused')} "
              f"({ok}/{len(RESULTS)} done) {rec.get('error', '')[:120]}",
              flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    summ = json.load(open(os.path.join(ROOT,
                                       "opion/data/rl/iter_0002_summary.json")))
    sandbox_ids = [r["sandbox"] for r in summ["sandboxes"]]
    assert len(sandbox_ids) == NSB
    jobs = build_jobs()
    chunks = [[] for _ in range(NSB)]
    for j in jobs:  # interleave pairs across sandboxes
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
    print(f"[pairs] {NSB} sandboxes, {len(jobs)} matches total", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            print(f"[pairs monitor] t={(time.monotonic()-t0)/60:.1f}min "
                  f"done={len(RESULTS)}/8", flush=True)
    for th in ths:
        th.join()

    agg = {}
    bad = 0
    for k in sorted(RESULTS):
        for row in RESULTS[k].get("rows", []):
            if row["rc"] != 0 or row.get("net_x") is None:
                bad += 1
                continue
            key = f"{row['x']} vs {row['y']}"
            agg.setdefault(key, []).append(row["net_x"] / 400.0)
    out = {"per_match": {f"sb{k}": RESULTS[k].get("rows", [])
                         for k in sorted(RESULTS)}}
    with open(os.path.join(ROOT, "opion/data/rl/pairs_summary.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    print("\n=== ROUND-ROBIN (net of FIRST-listed bot, stacks/hand, "
          "mirror legs) ===", flush=True)
    for key, vals in agg.items():
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        se = (var / len(vals)) ** 0.5
        print(f"  {key:<12} {mean:+.4f} ± {se:.4f}  "
              f"({len(vals)} matches)", flush=True)
    print(f"[pairs] bad matches: {bad}", flush=True)
    print("PAIRS_DONE", flush=True)


if __name__ == "__main__":
    main()
