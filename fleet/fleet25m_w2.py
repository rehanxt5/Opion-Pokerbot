#!/usr/bin/env python3
"""Wave-2 top-up: 20 fresh sandboxes x ~520K teacher decisions to close the
gap from wave 1 (14.95M) to the 25M target. Lessons from wave 1 baked in:
- create-retry (paused sandboxes / transient 429s)
- wait for run_shard FULL exit ([r] pgrep trick) before tar (no gzip race)
- tar 'data' only (no top-level logs dir), pull with retries + semaphore
- pause (never kill) after verified pull; pause-lifecycle on create too.
Seeds are 'w2:{sb}:r{round}:m{i}' — no collision with wave 1."""
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "fleet25m", "wave2")
NSB = 20
QUOTA = 520_000
WATCHDOG = 2400
WAVE = 2
PULL_SEM = threading.Semaphore(6)

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
        last = None
        for attempt in range(8):  # create-retry: transient 429s
            try:
                sb = Sandbox.create(template=tpl, timeout=3600,
                                    metadata={"wave": str(WAVE), "sb": str(idx)},
                                    lifecycle=SandboxLifecycle(on_timeout="pause",
                                                               auto_resume=True))
                break
            except Exception as e:
                last = e
                time.sleep(90)
        if sb is None:
            raise RuntimeError(f"create failed after retries: {repr(last)[:160]}")
        rec["create_s"] = round(time.monotonic() - t0, 1)
        rec["sandbox"] = sb.sandbox_id
        with LOCK:
            print(f"[{tag}] up ({rec['create_s']}s) {sb.sandbox_id}", flush=True)
        cmd = (f"cd /home/user/opion && python3 run_shard.py --bot bot/elite_instr.py "
               f"--teacher-name elitev6 --engine engine_log.py --opponents pairs.json "
               f"--quota {QUOTA} --matches 8 --hands 400 --stage 1 --wave {WAVE} "
               f"--sb {idx} --out-dir /home/user/opion/data --gzip --timeout-s {WATCHDOG}")
        t1 = time.monotonic()
        r = sb.commands.run(cmd, timeout=WATCHDOG + 1200)
        rec["run_s"] = round(time.monotonic() - t1, 1)
        rec["rc"] = r.exit_code
        # wait for run_shard FULL exit (incl. gzip pass) — no tar race this time
        for _ in range(40):
            q = sb.commands.run(
                "pgrep -f '[r]un_shard.py' >/dev/null && echo 1 || echo 0", timeout=60)
            if q.stdout.strip() == "0":
                break
            time.sleep(20)
        try:
            info = sb.files.read("/home/user/opion/data/summary.json")
            rec["summary"] = json.loads(info if isinstance(info, str) else info.decode())
        except Exception as e:
            rec["summary"] = {"error": repr(e)[:120]}
        t2 = time.monotonic()
        r = sb.commands.run("cd /home/user/opion && rm -f shard.tgz && "
                            "tar -czf shard.tgz data", timeout=900)
        rec["tar_rc"] = r.exit_code
        if r.exit_code != 0:
            raise RuntimeError(f"tar rc={r.exit_code}")
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)
        data = None
        for attempt in range(3):
            try:
                with PULL_SEM:
                    data = sb.files.read("/home/user/opion/shard.tgz", format="bytes")
                break
            except Exception as e:
                rec.setdefault("pull_retries", []).append(repr(e)[:80])
                time.sleep(45)
        if data is None:
            raise RuntimeError("pull failed after retries")
        assert data[:2] == b"\x1f\x8b", "bad gzip magic"
        with open(os.path.join(OUT, tag, "shard.tgz"), "wb") as f:
            f.write(data)
        rec["pull_mb"] = round(len(data) / 1e6, 1)
        rec["pull_s"] = round(time.monotonic() - t2, 1)
        rec["pulled"] = True
        sb.beta_pause()  # PAUSE — never kill (user directive)
        rec["paused"] = True
        s = rec.get("summary", {})
        with LOCK:
            print(f"[{tag}] DONE rc={rec.get('rc')} teacher={s.get('teacher_decisions', '?')} "
                  f"rounds={s.get('rounds', '?')} run={rec['run_s']}s "
                  f"pull={rec['pull_mb']}MB/{rec['pull_s']}s", flush=True)
    except Exception as e:
        rec["error"] = repr(e)[:300]
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
