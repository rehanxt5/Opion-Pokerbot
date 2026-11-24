#!/usr/bin/env python3
"""Multi-sandbox parallel test: 2 sandboxes from opion-gen-v1, 5,000 samples each
(10K total), run simultaneously; verifies disjoint seeds + per-sandbox speed."""
import glob
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "pilot_2x5k")

RESULTS = {}


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def run_one(idx, tpl, seed_base, quota):
    from e2b import Sandbox
    t0 = time.monotonic()
    sb = Sandbox.create(template=tpl, timeout=1800)
    info = {"sandbox": sb.sandbox_id, "create_s": round(time.monotonic() - t0, 1)}
    print(f"[sb{idx}] sandbox {sb.sandbox_id} up in {info['create_s']}s", flush=True)
    cmd = (f"cd /home/user/opion && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           f"python3 run_shard.py --bot bot/elite.py --engine engine_log.py --quota {quota} "
           f"--matches 8 --hands 400 --stage 1 --seed-base {seed_base} "
           f"--out-dir /home/user/opion/data")
    t1 = time.monotonic()
    r = sb.commands.run(cmd, timeout=900)
    info["run_s"] = round(time.monotonic() - t1, 1)
    info["rc"] = r.exit_code
    try:
        info["summary"] = json.loads(sb.files.read("/home/user/opion/data/summary.json"))
    except Exception as e:
        info["summary"] = {"error": repr(e)[:120]}
    os.makedirs(os.path.join(OUT, f"sb{idx}"), exist_ok=True)
    pulled = 0
    for i in range(8):
        try:
            data = sb.files.read(f"/home/user/opion/data/shard_m{i}.jsonl")
            if isinstance(data, str):
                data = data.encode()
            with open(os.path.join(OUT, f"sb{idx}", f"shard_m{i}.jsonl"), "wb") as f:
                f.write(data)
            pulled += 1
        except Exception as e:
            info.setdefault("pull_errs", []).append(repr(e)[:80])
    info["shards_pulled"] = pulled
    try:
        sb.kill()
    except Exception:
        pass
    RESULTS[idx] = info
    s = info["summary"]
    print(f"[sb{idx}] DONE rc={info['rc']} records={s.get('records_total', '?')} "
          f"dec/s={s.get('dec_per_s_all_cores', '?')} run={info['run_s']}s "
          f"shards={pulled}/8", flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    tpl = open(os.path.join(FLEET, "template.txt")).read().strip()
    t0 = time.monotonic()
    ths = [threading.Thread(target=run_one, args=(i, tpl, 770001 + i * 100000, 5000))
           for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - t0
    tot = sum(r.get("summary", {}).get("records_total", 0) for r in RESULTS.values())
    print(f"\ncombined: {tot} records, wall {wall:.1f}s -> {tot / wall:.0f} dec/s aggregate")
    seeds = [set(), set()]
    for i in range(2):
        for fp in glob.glob(os.path.join(OUT, f"sb{i}", "shard_m*.jsonl")):
            for line in open(fp, errors="replace"):
                if '"t":"deal"' in line:
                    seeds[i].add(json.loads(line)["seed"])
    print(f"deal seeds: sb0={len(seeds[0])} sb1={len(seeds[1])} overlap={len(seeds[0] & seeds[1])}")
    print("TEST2X5K_DONE")


if __name__ == "__main__":
    main()
