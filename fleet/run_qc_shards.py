#!/usr/bin/env python3
"""Per-shard QC: qc_report.py on one sandbox dir at a time (the monolithic
run OOM-died at 15M records). 3 workers, then aggregate per-shard verdicts
into one combined report."""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
BASE = os.path.join(ROOT, "opion", "data", "fleet25m")
PY = os.path.join(ROOT, ".venv", "bin", "python")
WORKERS = 3


def run_one(args):
    wave_dir, tag = args
    out_dir = os.path.join(BASE, f"qc_{tag.split('_')[0]}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{tag}.json")
    cmd = [PY, os.path.join(FLEET, "qc_report.py"),
           "--data-dir", os.path.join(BASE, wave_dir, tag),
           "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    ok = r.returncode == 0
    verdict = "?"
    if os.path.exists(out):
        try:
            verdict = json.load(open(out)).get("verdict", "?")
        except Exception:
            pass
    if not ok:
        with open(out + ".stderr", "w") as f:
            f.write(r.stderr[-4000:])
    return (tag, ok, verdict)


def main():
    jobs = []
    for wave_dir in ("wave1_data", "wave2_data"):
        for d in sorted(glob.glob(os.path.join(BASE, wave_dir, "sb*"))):
            tag = os.path.basename(d)
            jobs.append((wave_dir, tag))
    print(f"[qc] {len(jobs)} shard jobs, {WORKERS} workers", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(run_one, jobs)):
            results.append(res)
            ok, verdict = res[1], res[2]
            print(f"[qc] {i + 1}/{len(jobs)} {res[0]}: "
                  f"{'PASS' if ok else 'FAIL'} ({verdict})", flush=True)
    fails = [r[0] for r in results if not r[1]]
    agg = {"total_shards": len(results), "passed": len(results) - len(fails),
           "failed": fails,
           "shards": {r[0]: {"exit_ok": r[1], "verdict": r[2]}
                      for r in results}}
    with open(os.path.join(BASE, "qc_shards_summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print(f"QC_SHARDS_DONE passed={len(results) - len(fails)}/{len(results)} "
          f"fails={fails}", flush=True)


if __name__ == "__main__":
    main()
