#!/usr/bin/env python3
"""Wave-1 takeover: the original fleet25m orchestrator died right after
creating the 20 sandboxes (older build -> no metadata, default kill
lifecycle, ids only in /tmp/wave1_ids.json). This script reconnects to all
of them, re-arms the 1h session timeout, waits for run_shard's watchdog to
finish (summary.json appears), tars + pulls each shard, and then PAUSES
every sandbox (never kill — user directive)."""
import json
import os
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "fleet25m", "tookover_w1")
IDS_FILE = "/tmp/wave1_ids.json"
PULL_SEM = threading.Semaphore(5)
LOCK = threading.Lock()
RESULTS = {}
POLL_END = time.time() + 21 * 60  # hard stop waiting on summary.json (UTC)


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("E2B_KEY not found in .env")


def takeover(i, sid):
    tag = f"cand{i:02d}"  # provisional; true sb index recovered from seeds post-pull
    rec = {"sandbox": sid}
    sb = None
    try:
        from e2b import Sandbox
        sb = Sandbox.connect(sandbox_id=sid, timeout=3600)
        try:
            sb.set_timeout(3600)  # re-arm session deadline (default lifecycle kills!)
        except Exception as e:
            rec["set_timeout"] = repr(e)[:120]
        with LOCK:
            print(f"[{tag}] connected {sid} deadline re-armed", flush=True)
        got = False
        while time.time() < POLL_END:
            r = sb.commands.run(
                "test -f /home/user/opion/data/summary.json && echo 1 || echo 0",
                timeout=60)
            if r.stdout.strip() == "1":
                got = True
                break
            time.sleep(45)
        rec["summary_seen"] = got
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)
        try:
            info = sb.files.read("/home/user/opion/data/summary.json",
                                 format="bytes")
            with open(os.path.join(OUT, tag, "summary.json"), "wb") as f:
                f.write(info)
            rec["summary"] = json.loads(info.decode())
        except Exception as e:
            rec["summary_err"] = repr(e)[:120]
        t0 = time.monotonic()
        r = sb.commands.run("cd /home/user/opion && tar -czf shard.tgz data",
                            timeout=1200)
        rec["tar_rc"] = r.exit_code
        with PULL_SEM:
            data = sb.files.read("/home/user/opion/shard.tgz", format="bytes")
        rec["pull_mb"] = round(len(data) / 1e6, 1)
        rec["pull_s"] = round(time.monotonic() - t0, 1)
        assert data[:2] == b"\x1f\x8b", "bad gzip magic"
        with open(os.path.join(OUT, tag, "shard.tgz"), "wb") as f:
            f.write(data)
        rec["pulled"] = True
        sb.beta_pause()  # PAUSE — never kill (user directive)
        rec["paused"] = True
        s = rec.get("summary", {}) or {}
        with LOCK:
            print(f"[{tag}] PAUSED teacher={s.get('teacher_decisions', '?')} "
                  f"rounds={s.get('rounds', '?')} pull={rec['pull_mb']}MB/"
                  f"{rec['pull_s']}s", flush=True)
    except Exception as e:
        rec["error"] = repr(e)[:300]
        with LOCK:
            print(f"[{tag}] ERROR {rec['error']}", flush=True)
        if sb is not None:
            try:
                sb.beta_pause()  # even on failure: pause, never kill
                rec["paused"] = True
            except Exception:
                pass
    RESULTS[i] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    ids = json.load(open(IDS_FILE))
    os.makedirs(OUT, exist_ok=True)
    print(f"[takeover] reconnecting to {len(ids)} sandboxes", flush=True)
    t0 = time.monotonic()
    ths = []
    for i, sid in enumerate(ids):
        th = threading.Thread(target=takeover, args=(i, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.4)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            n = sum(1 for r in RESULTS.values())
        print(f"[takeover] t={(time.monotonic() - t0) / 60:.1f}min "
              f"finished={n}/{len(ids)}", flush=True)
    for th in ths:
        th.join()
    ok = [r for r in RESULTS.values() if r.get("pulled")]
    tot = sum((r.get("summary") or {}).get("teacher_decisions", 0)
              for r in RESULTS.values())
    errs = [r["sandbox"] for r in RESULTS.values() if "error" in r]
    with open(os.path.join(OUT, "takeover_summary.json"), "w") as f:
        json.dump({"pulled": len(ok), "total_teacher": tot, "errors": errs,
                   "sandboxes": [RESULTS[i] for i in sorted(RESULTS)]},
                  f, indent=1, default=str)
    print(f"TAKEOVER_DONE pulled={len(ok)}/{len(ids)} total_teacher={tot} "
          f"errors={errs} wall={(time.monotonic() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
