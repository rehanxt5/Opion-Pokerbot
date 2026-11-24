#!/usr/bin/env python3
"""Census wave 2: run census.py on all 20 wave-2 sandboxes.

Same procedure as wave 1 (pause, never kill; census.json stays on disk):
  connect (auto-resume) -> upload census.py -> run (8 workers, ~seconds)
  -> verify CENSUS_OK -> pull back ONLY the trimmed summary -> pause.
Wave-2 list position NN maps to packed shard w2/sbNN; each result is
cross-checked against the packed manifest total.
"""
import json
import os
import sys
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
WAVE = 2
CONC = 6
IDS_FILE = "/tmp/wave2_ids.json"

RESULTS = {}
LOCK = threading.Lock()


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def one(idx, sid):
    tag = f"w2_{idx:02d}"
    rec = {"tag": tag, "sandbox": sid}
    from e2b import Sandbox
    sb = None
    try:
        last = None
        for attempt in range(4):
            try:
                sb = Sandbox.connect(sid, timeout=300)  # auto-resume
                break
            except Exception as e:
                last = e
                time.sleep(30 * (attempt + 1))
        if sb is None:
            raise RuntimeError(f"connect failed: {last!r}")
        sb.set_timeout(3600)
        with open(os.path.join(FLEET, "census.py"), "rb") as f:
            sb.files.write("/home/user/opion/census.py", f.read())
        t0 = time.monotonic()
        r = sb.commands.run(
            "cd /home/user/opion && python3 census.py "
            "--data-dir /home/user/opion/data "
            "--out /home/user/opion/census.json --workers 8", timeout=600)
        rec["run_s"] = round(time.monotonic() - t0, 1)
        rec["exit"] = r.exit_code
        if r.exit_code != 0 or "CENSUS_OK" not in (r.stdout or ""):
            raise RuntimeError("census rc=%s %s" % (r.exit_code,
                                                    (r.stderr or "")[-300:]))
        try:
            q = sb.commands.run(
                "python3 -c \"import json;print(json.load(open"
                "('/home/user/opion/data/summary.json'))['sb'])\"",
                timeout=60)
            rec["sb_arg"] = int(q.stdout.strip())
        except Exception:
            rec["sb_arg"] = None
        raw = sb.files.read("/home/user/opion/census.json")
        obj = json.loads(raw if isinstance(raw, str) else raw.decode())
        rec["totals"] = obj["totals"]
        rec["opponents"] = obj["opponents"]
        mpath = os.path.join(ROOT, "opion", "data", "packed", "w2",
                             f"sb{idx:02d}", "manifest.json")
        m = json.load(open(mpath))
        rec["manifest_total"] = m["total"]
        rec["delta_vs_manifest"] = obj["totals"]["teacher_records"] - m["total"]
        rec["ok"] = True
    except Exception as e:
        rec["error"] = repr(e)[:300]
        rec["ok"] = False
    if sb is not None:
        for attempt in range(4):
            try:
                sb.beta_pause()
                rec["paused"] = True
                break
            except Exception:
                time.sleep(20 * (attempt + 1))
        else:
            rec["paused"] = False
    with LOCK:
        RESULTS[idx] = rec
        ok = sum(1 for x in RESULTS.values() if x.get("ok"))
        print(f"[w2] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"run={rec.get('run_s', '?')}s paused={rec.get('paused')} "
              f"sb_arg={rec.get('sb_arg')} delta={rec.get('delta_vs_manifest')} "
              f"({ok}/{len(RESULTS)} done)", flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    ids = json.load(open(IDS_FILE))
    assert len(ids) == 20, f"expected 20 wave-2 ids, got {len(ids)}"
    outdir = os.path.join(ROOT, "opion", "data", "census")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.monotonic()
    sem = threading.Semaphore(CONC)

    def worker(i, sid):
        with sem:
            one(i, sid)

    ths = []
    for i, sid in enumerate(ids):
        th = threading.Thread(target=worker, args=(i, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(2)  # stagger resumes (429 accounting lag)
    print(f"[w2] census wave 2: {len(ids)} sandboxes, conc={CONC}", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(120)  # 2-minute monitoring cadence (user request)
        with LOCK:
            n_ok = sum(1 for x in RESULTS.values() if x.get("ok"))
            n_err = len(RESULTS) - n_ok
            n_paused = sum(1 for x in RESULTS.values() if x.get("paused"))
        print(f"[w2 monitor] t={(time.monotonic() - t0) / 60:.1f}min "
              f"ok={n_ok} err={n_err} paused={n_paused} /{len(ids)}",
              flush=True)
    for th in ths:
        th.join()

    for i in sorted(RESULTS):
        r = RESULTS[i]
        if r.get("ok"):
            with open(os.path.join(outdir, f"w2_sb{i:02d}_census.json"),
                      "w") as f:
                json.dump({"tag": r["tag"], "sandbox": r["sandbox"],
                           "sb_arg": r.get("sb_arg"),
                           "totals": r["totals"],
                           "opponents": r["opponents"]}, f, indent=1)
    tot = sum(r["totals"]["teacher_records"] for r in RESULTS.values()
              if r.get("ok"))
    man = sum(r["manifest_total"] for r in RESULTS.values() if r.get("ok"))
    errs = [r["tag"] for r in RESULTS.values() if not r.get("ok")]
    nopaused = [r["tag"] for r in RESULTS.values() if not r.get("paused")]
    out = {"wave": 2, "ok": len(ids) - len(errs), "errors": errs,
           "not_paused": nopaused, "census_teacher_total": tot,
           "manifest_total": man, "delta": tot - man,
           "sandboxes": [RESULTS[i] for i in sorted(RESULTS)]}
    with open(os.path.join(outdir, "census_w2_result.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[w2] census_teacher_total={tot:,} vs manifest={man:,} "
          f"(delta {tot - man:+,})", flush=True)
    print(f"CENSUS_W2_DONE ok={len(ids) - len(errs)}/{len(ids)} "
          f"errors={errs} not_paused={nopaused}", flush=True)


if __name__ == "__main__":
    main()
