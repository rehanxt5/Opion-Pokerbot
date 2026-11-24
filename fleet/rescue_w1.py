#!/usr/bin/env python3
"""Wave-1 rescue: reconnect to the 20 running sandboxes, wait for their
run_shard watchdog/quota to finish, tar+pull all data, then PAUSE (not kill).
Sandbox ids come from /tmp/fleet_w1.log ('[sbNN] up (...) <id>' lines)."""
import glob
import json
import os
import re
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "fleet25m")
MAX_WAIT_MIN = 55

RESULTS = {}
LOCK = threading.Lock()


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def ids_from_log():
    pairs = []
    for line in open("/tmp/fleet_w1.log"):
        m = re.match(r"\[sb(\d+)\] up \([\d.]+s\) (\S+)", line.strip())
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return sorted(pairs)


def rescue(idx, sid):
    tag = f"sb{idx:02d}"
    rec = {"tag": tag, "sandbox": sid}
    sb = None
    try:
        from e2b import Sandbox
        sb = Sandbox.connect(sandbox_id=sid, timeout=3600)
        with LOCK:
            print(f"[{tag}] connected", flush=True)
        # wait for run_shard to finish (summary.json written at the end)
        waited = 0
        while waited < MAX_WAIT_MIN * 60:
            r = sb.commands.run(
                "test -f /home/user/opion/data/summary.json && echo 1 || echo 0",
                timeout=120)
            if r.stdout.strip() == "1":
                break
            time.sleep(60)
            waited += 60
        rec["waited_min"] = round(waited / 60, 1)
        r = sb.commands.run("cd /home/user/opion && tar -czf shard.tgz data logs",
                            timeout=900)
        rec["tar_rc"] = r.exit_code
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)
        data = sb.files.read("/home/user/opion/shard.tgz", format="bytes")
        with open(os.path.join(OUT, tag, "shard.tgz"), "wb") as f:
            f.write(data)
        rec["pull_mb"] = round(len(data) / 1e6, 1)
        info = sb.files.read("/home/user/opion/data/summary.json", format="bytes")
        with open(os.path.join(OUT, tag, "summary.json"), "wb") as f:
            f.write(info)
        rec["summary"] = json.loads(info if isinstance(info, str) else info.decode())
        sb.beta_pause()  # PAUSE — sandbox state preserved, no data loss
        rec["paused"] = True
        s = rec.get("summary", {})
        with LOCK:
            print(f"[{tag}] PAUSED teacher={s.get('teacher_decisions', '?')} "
                  f"pull={rec['pull_mb']}MB waited={rec['waited_min']}min", flush=True)
    except Exception as e:
        rec["error"] = repr(e)[:300]
        with LOCK:
            print(f"[{tag}] ERROR {rec['error']}", flush=True)
        if sb is not None:
            try:
                sb.beta_pause()
            except Exception:
                pass
    RESULTS[idx] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    pairs = ids_from_log()
    print(f"rescuing {len(pairs)} sandboxes", flush=True)
    t0 = time.monotonic()
    ths = []
    for idx, sid in pairs:
        th = threading.Thread(target=rescue, args=(idx, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.3)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        el = (time.monotonic() - t0) / 60
        with LOCK:
            n = sum(1 for r in RESULTS.values())
        print(f"[rescue] t={el:.1f}min finished={n}/{len(pairs)}", flush=True)
    for th in ths:
        th.join()
    tot = sum(r.get("summary", {}).get("teacher_decisions", 0)
              for r in RESULTS.values() if "summary" in r)
    errs = [r["tag"] for r in RESULTS.values() if "error" in r]
    with open(os.path.join(OUT, "rescue_summary.json"), "w") as f:
        json.dump({"total_teacher": tot, "errors": errs,
                   "sandboxes": [RESULTS[i] for i in sorted(RESULTS)]}, f, indent=1, default=str)
    print(f"RESCUE_DONE total_teacher={tot} errors={errs} "
          f"wall={(time.monotonic() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
