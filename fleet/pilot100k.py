#!/usr/bin/env python3
"""100K diverse pilot: one sandbox from opion-gen-v1, 100K teacher decisions
across the full opponent mix; pulls everything back for the local QC gate."""
import os
import sys
import tarfile
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "pilot100k")


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
    print(f"sandbox {sb.sandbox_id} up in {time.monotonic() - t0:.1f}s")

    cmd = ("cd /home/user/opion && python3 run_shard.py --bot bot/elite_instr.py "
           "--teacher-name elitev6 --engine engine_log.py --opponents pairs.json "
           "--quota 100000 --matches 8 --hands 400 --stage 1 --wave 1 --sb 0 "
           "--out-dir /home/user/opion/data --gzip --timeout-s 2400")
    t1 = time.monotonic()
    r = sb.commands.run(cmd, timeout=3000)
    print(f"run rc={r.exit_code} wall={time.monotonic() - t1:.1f}s")
    print((r.stdout or "")[-1800:])
    if r.exit_code != 0:
        print("STDERR", (r.stderr or "")[-1200:])

    # pull one archive: data + logs (soft sidecars included)
    t2 = time.monotonic()
    r = sb.commands.run("cd /home/user/opion && tar -czf pilot.tgz data", timeout=600)
    print("tar rc", r.exit_code)
    os.makedirs(OUT, exist_ok=True)
    tgz_path = os.path.join(OUT, "pilot.tgz")
    data = sb.files.read("/home/user/opion/pilot.tgz", format="bytes")
    with open(tgz_path, "wb") as f:
        f.write(data)
    print(f"pulled {tgz_path} {os.path.getsize(tgz_path)/1e6:.1f}MB "
          f"in {time.monotonic() - t2:.1f}s")
    with tarfile.open(tgz_path) as tf:
        tf.extractall(OUT)
    try:
        sb.kill()
    except Exception:
        pass
    print("PILOT100K_DONE data at", os.path.join(OUT, "data"))


if __name__ == "__main__":
    main()
