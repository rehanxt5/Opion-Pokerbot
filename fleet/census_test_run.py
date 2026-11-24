#!/usr/bin/env python3
"""Census test run on the test sandbox: upload census.py, run it (8 workers),
read back the small result JSON for verification, cross-check against the
packed w1/sb00 manifest, then pause the test sandbox (never kill).
The census.json stays on the sandbox disk either way."""
import json
import os
import sys
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox

    ids = json.load(open("/tmp/census_test_ids.json"))
    sb = Sandbox.connect(ids["test"], timeout=300)
    sb.set_timeout(3600)
    print(f"[run] connected {sb.sandbox_id}", flush=True)

    with open(os.path.join(FLEET, "census.py"), "rb") as f:
        sb.files.write("/home/user/opion/census.py", f.read())
    print("[run] census.py uploaded", flush=True)

    t0 = time.monotonic()
    r = sb.commands.run(
        "cd /home/user/opion && python3 census.py "
        "--data-dir /home/user/opion/data --out /home/user/opion/census.json "
        "--workers 8", timeout=900)
    wall = time.monotonic() - t0
    print(r.stdout, flush=True)
    if r.stderr:
        print("[run] stderr tail:", r.stderr[-800:], flush=True)
    print(f"[run] exit={r.exit_code} wall={wall:.1f}s", flush=True)
    if r.exit_code != 0:
        sys.exit("CENSUS_RUN_FAILED")

    # verification read-back (the file also stays on the sandbox disk)
    raw = sb.files.read("/home/user/opion/census.json")
    obj = json.loads(raw if isinstance(raw, str) else raw.decode())
    t = obj["totals"]

    m = json.load(open(os.path.join(ROOT, "opion", "data", "packed", "w1",
                                    "sb00", "manifest.json")))
    print("[check] census vs packed manifest (source of w1/sb00 data):",
          flush=True)
    print(f"  teacher_records: {t['teacher_records']:,} vs {m['total']:,} "
          f"(delta {t['teacher_records'] - m['total']:+,})", flush=True)
    print(f"  teacher_act:     {t['teacher_act']:,} vs {m['teacher_act']:,} "
          f"(delta {t['teacher_act'] - m['teacher_act']:+,})", flush=True)
    print(f"  teacher_disc:    {t['teacher_disc']:,} vs {m['teacher_disc']:,} "
          f"(delta {t['teacher_disc'] - m['teacher_disc']:+,})", flush=True)
    print(f"  matches:         {t['files']:,} vs {m['matches']:,}", flush=True)
    ok = (t["teacher_records"] >= m["total"]
          and t["teacher_act"] >= m["teacher_act"]
          and abs(t["teacher_records"] - m["total"]) <= 0.005 * m["total"]
          and not t["anomalies"])
    print("[check] " + ("CONSISTENT" if ok else "MISMATCH — investigate"),
          flush=True)

    try:
        sb.beta_pause()
        print("[run] test sandbox paused (census.json left on its disk)",
              flush=True)
    except Exception as e:
        print(f"[run] pause note: {e!r}", flush=True)
    print("CENSUS_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
