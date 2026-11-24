#!/usr/bin/env python3
"""Download packed Phase-2 data (acts.npz, disc.npz, manifest.json) from the
40 dataset sandboxes (wave 1 + wave 2; test sandbox excluded), then pause
each sandbox. Saves to opion/data/packed/{w1,w2}/sbNN/."""
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "packed")
SEM = threading.Semaphore(8)
LOCK = threading.Lock()
RESULTS = {}


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def pull_one(wave, idx, sid):
    tag = f"{wave}/sb{idx:02d}"
    rec = {}
    sb = None
    try:
        from e2b import Sandbox
        t0 = time.monotonic()
        sb = Sandbox.connect(sandbox_id=sid, timeout=3600)
        try:
            sb.set_timeout(3600)
        except Exception:
            pass
        d = os.path.join(OUT, wave, f"sb{idx:02d}")
        os.makedirs(d, exist_ok=True)
        with SEM:
            for fn in ("acts.npz", "disc.npz", "manifest.json"):
                data = sb.files.read(f"/home/user/opion/packed/{fn}",
                                     format="bytes")
                if fn.endswith(".npz") and data[:2] != b"PK":
                    raise RuntimeError(f"{fn}: bad npz magic {data[:2]!r}")
                with open(os.path.join(d, fn), "wb") as f:
                    f.write(data)
        man = json.load(open(os.path.join(d, "manifest.json")))
        rec.update(total=man["total"], mb=round(sum(
            os.path.getsize(os.path.join(d, f))
            for f in ("acts.npz", "disc.npz")) / 1e6, 1),
            wall_s=round(time.monotonic() - t0, 1))
        rec["ok"] = not man.get("hard_fail")
        sb.beta_pause()  # pause right after verified pull (never kill)
        rec["paused"] = True
        with LOCK:
            print(f"[{tag}] ok={rec['ok']} total={rec['total']} "
                  f"{rec['mb']}MB {rec['wall_s']}s -> PAUSED", flush=True)
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
    RESULTS[(wave, idx)] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    jobs = []
    for wave, idfile in (("w1", "/tmp/wave1_ids.json"),
                         ("w2", "/tmp/wave2_ids.json")):
        for i, sid in enumerate(json.load(open(idfile))):
            jobs.append((wave, i, sid))
    print(f"[dl] pulling packed data from {len(jobs)} sandboxes", flush=True)
    t0 = time.monotonic()
    ths = []
    for wave, i, sid in jobs:
        th = threading.Thread(target=pull_one, args=(wave, i, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.3)
    while any(th.is_alive() for th in ths):
        time.sleep(30)
        with LOCK:
            done = sum(1 for r in RESULTS.values() if r)
        print(f"[dl] t={(time.monotonic() - t0) / 60:.1f}min done={done}/{len(jobs)}",
              flush=True)
    for th in ths:
        th.join()
    tot = sum(r.get("total", 0) for r in RESULTS.values() if r.get("ok"))
    mb = sum(r.get("mb", 0) for r in RESULTS.values() if r.get("ok"))
    bad = [k for k, r in RESULTS.items() if not r.get("ok")]
    with open(os.path.join(OUT, "download_summary.json"), "w") as f:
        json.dump({f"{w}/sb{i:02d}": r for (w, i), r in sorted(RESULTS.items())},
                  f, indent=1, default=str)
    print(f"DOWNLOAD_DONE ok_total={tot:,} mb={mb:.0f} failed={bad} "
          f"wall={(time.monotonic() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
