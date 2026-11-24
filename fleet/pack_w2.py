#!/usr/bin/env python3
"""Wave-2 pack rollout: resume each wave-1 sandbox, upload spec.py +
pack_batch.py, pack its shard on 8 cores, verify PACK_OK + clean manifest,
then PAUSE it immediately (never kill). Packed data stays on the sandbox.
Progress prints every minute; each sandbox pauses the moment it is done."""
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
IDS_FILE = "/tmp/wave2_ids.json"
LOCK = threading.Lock()
RESULTS = {}


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def pack_one(idx, sid):
    tag = f"sb{idx:02d}"
    rec = {"sandbox": sid}
    sb = None
    try:
        from e2b import Sandbox
        t0 = time.monotonic()
        sb = Sandbox.connect(sandbox_id=sid, timeout=3600)
        try:
            sb.set_timeout(3600)
        except Exception:
            pass
        for fn in ("spec.py", "pack_batch.py"):
            with open(os.path.join(FLEET, fn), "rb") as f:
                sb.files.write(f"/home/user/opion/{fn}", f.read())
        rec["resume_s"] = round(time.monotonic() - t0, 1)
        t1 = time.monotonic()
        r = sb.commands.run(
            "cd /home/user/opion && rm -rf packed && "
            "python3 pack_batch.py --data-dir data --out-dir packed "
            "--workers 8 > pack.log 2>&1; tail -c 2000 pack.log",
            timeout=1200)
        rec["pack_s"] = round(time.monotonic() - t1, 1)
        rec["rc"] = r.exit_code
        out = r.stdout or ""
        rec["pack_ok"] = "PACK_OK" in out
        try:
            m = sb.files.read("/home/user/opion/packed/manifest.json")
            man = json.loads(m if isinstance(m, str) else m.decode())
            rec["total"] = man.get("total")
            rec["hard_fail"] = man.get("hard_fail")
            rec["bytes_per_dec"] = man.get("bytes_per_decision")
            rec["pack_ok"] = rec["pack_ok"] and not man.get("hard_fail")
        except Exception as e:
            rec["manifest_err"] = repr(e)[:120]
            rec["pack_ok"] = False
        sb.beta_pause()  # PAUSE the moment work is verified (never kill)
        rec["paused"] = True
        rec["wall_s"] = round(time.monotonic() - t0, 1)
        with LOCK:
            print(f"[{tag}] DONE ok={rec['pack_ok']} total={rec.get('total')} "
                  f"b/dec={rec.get('bytes_per_dec')} "
                  f"pack={rec['pack_s']}s wall={rec['wall_s']}s -> PAUSED",
                  flush=True)
    except Exception as e:
        rec["error"] = repr(e)[:250]
        with LOCK:
            print(f"[{tag}] ERROR {rec['error']}", flush=True)
        if sb is not None:
            try:
                sb.beta_pause()
                rec["paused"] = True
            except Exception:
                pass
    RESULTS[idx] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    ids = json.load(open(IDS_FILE))
    print(f"[pack-w2] rolling out to {len(ids)} wave-1 sandboxes", flush=True)
    t0 = time.monotonic()
    ths = []
    for i, sid in enumerate(ids):
        th = threading.Thread(target=pack_one, args=(i, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.5)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            done = sum(1 for r in RESULTS.values())
            ok = sum(1 for r in RESULTS.values() if r.get("pack_ok"))
        print(f"[pack-w2] t={(time.monotonic() - t0) / 60:.1f}min "
              f"finished={done}/{len(ids)} pack_ok={ok}", flush=True)
    for th in ths:
        th.join()
    ok = [f"sb{i:02d}" for i, r in RESULTS.items() if r.get("pack_ok")]
    bad = [f"sb{i:02d}" for i, r in RESULTS.items() if not r.get("pack_ok")]
    with open(os.path.join(FLEET, "pack_w2_result.json"), "w") as f:
        json.dump({f"sb{i:02d}": r for i, r in sorted(RESULTS.items())},
                  f, indent=1, default=str)
    print(f"PACK_W2_DONE ok={len(ok)} failed={bad} "
          f"wall={(time.monotonic() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
