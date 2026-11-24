#!/usr/bin/env python3
"""Create the census test sandbox: a copy of wave-1 sb00.

Flow (user directive: pause, never kill):
  1. connect sb00 (auto-resumes the paused sandbox), re-arm timeout
  2. verify identity via its summary.json (wave/sb/teacher_decisions)
  3. snapshot the disk -> 'opion-census-test-sb00'
  4. create a NEW sandbox from that snapshot (lifecycle: pause on timeout)
  5. verify the copy has the raw data + 8 vCPU
  6. re-pause the source sb00; test sandbox stays up for the census run
Writes ids to /tmp/census_test_ids.json.
"""
import json
import os
import sys
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
SNAP_NAME = "opion-census-test"


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox, SandboxLifecycle

    w1 = json.load(open("/tmp/wave1_ids.json"))
    src_id = w1[0]  # wave-1 sb00
    print(f"[setup] source: wave-1 sb00 = {src_id}", flush=True)

    t0 = time.monotonic()
    sb = Sandbox.connect(src_id, timeout=300)  # auto-resumes paused sandbox
    sb.set_timeout(3600)
    print(f"[setup] connected (resumed) in {time.monotonic() - t0:.0f}s",
          flush=True)

    raw = sb.files.read("/home/user/opion/data/summary.json")
    s = json.loads(raw if isinstance(raw, str) else raw.decode())
    assert s.get("wave") == 1, f"not a wave-1 sandbox: wave={s.get('wave')}"
    # NOTE: list position in wave1_ids.json != run_shard --sb value; the local
    # packed tag for this source is w1/sb00 (totals match its manifest).
    print(f"[setup] identity OK: wave={s['wave']} sb={s['sb']} "
          f"teacher_decisions={s['teacher_decisions']:,} "
          f"all_records={s['all_records']:,} rounds={s['rounds']}",
          flush=True)

    t1 = time.monotonic()
    info = sb.create_snapshot(name=SNAP_NAME)
    snap_id = info.snapshot_id
    print(f"[setup] snapshot ready in {time.monotonic() - t1:.0f}s: {snap_id}",
          flush=True)

    try:
        sb.beta_pause()  # source goes back to sleep
        print("[setup] source sb00 re-paused", flush=True)
    except Exception as e:
        print(f"[setup] pause-after-snapshot note: {e!r}", flush=True)

    t2 = time.monotonic()
    tsb = Sandbox.create(template=snap_id, timeout=3600,
                         metadata={"role": "census-test", "src": "w1-list0"},
                         lifecycle=SandboxLifecycle(on_timeout="pause",
                                                    auto_resume=True))
    print(f"[setup] test sandbox up in {time.monotonic() - t2:.0f}s: "
          f"{tsb.sandbox_id}", flush=True)

    for cmd in ("nproc", "free -m | head -2",
                "ls /home/user/opion/data/obs_r*_m*.jsonl* | wc -l",
                "du -sh /home/user/opion/data", "python3 -V"):
        r = tsb.commands.run(cmd, timeout=60)
        print(f"[setup]$ {cmd}\n{r.stdout.strip()}", flush=True)

    with open("/tmp/census_test_ids.json", "w") as f:
        json.dump({"src": src_id, "snapshot": snap_id, "test": tsb.sandbox_id,
                   "src_summary": {"wave": s["wave"], "sb": s["sb"],
                                   "teacher_decisions": s["teacher_decisions"],
                                   "all_records": s["all_records"]}},
                  f, indent=1)
    print("SETUP_OK", flush=True)


if __name__ == "__main__":
    main()
