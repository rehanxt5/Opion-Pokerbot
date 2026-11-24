#!/usr/bin/env python3
"""Create ONE sandbox from the opion-gen-v1 snapshot, generate >=10K samples
with 8 parallel self-play matches, pull the shards back, report the speed."""
import json
import os
import sys
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "pilot10k")


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox

    tpl = open(os.path.join(FLEET, "template.txt")).read().strip()
    t0 = time.monotonic()
    sb = Sandbox.create(template=tpl, timeout=3600)
    print(f"sandbox {sb.sandbox_id} from snapshot in {time.monotonic() - t0:.1f}s")

    r = sb.commands.run(
        "nproc && free -m | head -2 && python3 -c 'import numpy; print(numpy.__version__)'",
        timeout=120)
    print(r.stdout.strip())

    cmd = ("cd /home/user/opion && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           "python3 run_shard.py --bot bot/elite.py --engine engine_log.py --quota 10000 "
           "--matches 8 --hands 400 --stage 1 --seed-base 770001 "
           "--out-dir /home/user/opion/data")
    t1 = time.monotonic()
    r = sb.commands.run(cmd, timeout=1500)
    print(f"run rc={r.exit_code} wall={time.monotonic() - t1:.1f}s")
    print((r.stdout or "")[-2500:])
    if r.exit_code != 0:
        print("STDERR", (r.stderr or "")[-1500:])

    os.makedirs(OUT, exist_ok=True)
    names = ["summary.json"] + [f"shard_m{i}.jsonl" for i in range(8)]
    for n in names:
        try:
            data = sb.files.read(f"/home/user/opion/data/{n}")
            if isinstance(data, str):
                data = data.encode()
            p = os.path.join(OUT, n)
            with open(p, "wb") as f:
                f.write(data)
            print("pulled", n, os.path.getsize(p), "bytes")
        except Exception as e:
            print("pull skip", n, repr(e)[:120])
    try:
        sb.kill()
    except Exception:
        pass
    print("TEST10K_DONE")


if __name__ == "__main__":
    main()
