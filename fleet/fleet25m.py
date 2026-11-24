#!/usr/bin/env python3
"""25M dataset wave (v2): 20 sandboxes x 1.25M teacher decisions, heterogeneous
opponent mix (pairs.json), instrumented teacher soft labels, per-sandbox archive
pull. Teacher rate ~280-400 dec/s per sandbox → each sandbox watchdogs at 45 min
(session cap 1h); waves top up to the 25M target.
"""
import glob
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "fleet25m")
NSB = 20
QUOTA = 1_250_000
WATCHDOG = 2400  # 40 min generation hard-stop; pull happens after (session cap 1h)
WAVE = 1

RESULTS = {}
LOCK = threading.Lock()


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def run_one(idx, tpl):
    tag = f"sb{idx:02d}"
    rec = {"tag": tag, "sb": idx}
    sb = None
    try:
        from e2b import Sandbox, SandboxLifecycle
        t0 = time.monotonic()
        # pause (not kill) on timeout/disconnect -> zero data loss (user directive)
        sb = Sandbox.create(template=tpl, timeout=3600,
                            metadata={"wave": str(WAVE), "sb": str(idx)},
                            lifecycle=SandboxLifecycle(on_timeout="pause",
                                                       auto_resume=True))
        rec["create_s"] = round(time.monotonic() - t0, 1)
        rec["sandbox"] = sb.sandbox_id
        with LOCK:
            print(f"[{tag}] up ({rec['create_s']}s) {sb.sandbox_id}", flush=True)
        cmd = (f"cd /home/user/opion && python3 run_shard.py --bot bot/elite_instr.py "
               f"--teacher-name elitev6 --engine engine_log.py --opponents pairs.json "
               f"--quota {QUOTA} --matches 8 --hands 400 --stage 1 --wave {WAVE} "
               f"--sb {idx} --out-dir /home/user/opion/data --gzip --timeout-s {WATCHDOG}")
        t1 = time.monotonic()
        r = sb.commands.run(cmd, timeout=WATCHDOG + 900)
        rec["run_s"] = round(time.monotonic() - t1, 1)
        rec["rc"] = r.exit_code
        try:
            info = sb.files.read("/home/user/opion/data/summary.json")
            rec["summary"] = json.loads(info if isinstance(info, str) else info.decode())
        except Exception as e:
            rec["summary"] = {"error": repr(e)[:120]}
        t2 = time.monotonic()
        r = sb.commands.run("cd /home/user/opion && tar -czf shard.tgz data logs", timeout=600)
        rec["tar_rc"] = r.exit_code
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)
        data = sb.files.read("/home/user/opion/shard.tgz", format="bytes")
        with open(os.path.join(OUT, tag, "shard.tgz"), "wb") as f:
            f.write(data)
        rec["pull_mb"] = round(len(data) / 1e6, 1)
        rec["pull_s"] = round(time.monotonic() - t2, 1)
        s = rec.get("summary", {})
        with LOCK:
            print(f"[{tag}] DONE rc={rec.get('rc')} teacher={s.get('teacher_decisions', '?')} "
                  f"rounds={s.get('rounds', '?')} run={rec['run_s']}s "
                  f"pull={rec['pull_mb']}MB/{rec['pull_s']}s", flush=True)
    except Exception as e:
        rec["error"] = repr(e)[:300]
        with LOCK:
            print(f"[{tag}] ERROR {rec['error']}", flush=True)
    finally:
        # PAUSE (not kill) — sandbox state preserved for re-pull if archive is bad
        if sb is not None:
            try:
                sb.beta_pause()
            except Exception:
                pass
    RESULTS[idx] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    tpl = open(os.path.join(FLEET, "template.txt")).read().strip()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.monotonic()
    ths = []
    for i in range(NSB):
        th = threading.Thread(target=run_one, args=(i, tpl), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.5)
    print(f"[fleet] wave {WAVE}: {NSB} sandboxes, quota {QUOTA} teacher decisions each "
          f"(target {NSB * QUOTA / 1e6:.1f}M), watchdog {WATCHDOG}s", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        el = (time.monotonic() - t0) / 60
        with LOCK:
            n = sum(1 for r in RESULTS.values())
        print(f"[fleet] t={el:.1f}min finished={n}/{NSB}", flush=True)
    for th in ths:
        th.join()
    wall = time.monotonic() - t0
    tot = sum(r.get("summary", {}).get("teacher_decisions", 0)
              for r in RESULTS.values() if "summary" in r)
    errs = [r["tag"] for r in RESULTS.values() if "error" in r]
    out = {"wave": WAVE, "total_teacher_decisions": tot,
           "wall_min": round(wall / 60, 1), "errors": errs,
           "sandboxes": [RESULTS[i] for i in sorted(RESULTS)]}
    with open(os.path.join(OUT, f"fleet_summary_w{WAVE}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"FLEET_RESULT wave={WAVE} total_teacher={tot} wall={wall / 60:.1f}min "
          f"errors={errs}", flush=True)
    print("FLEET25M_DONE", flush=True)


if __name__ == "__main__":
    main()
