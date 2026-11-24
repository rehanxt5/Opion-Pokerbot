#!/usr/bin/env python3
"""Wave-1 recovery for the 15 sandboxes whose takeover tar raced run_shard's
final gzip pass (summary.json is written BEFORE gzip, so the poll released
them while data/ was still changing; the except path paused them mid-pass).
Reconnect (auto-resumes), wait for run_shard to fully exit, re-tar cleanly,
pull with retries, then PAUSE again (never kill)."""
import json
import os
import threading
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FLEET))
OUT = os.path.join(ROOT, "opion", "data", "fleet25m", "tookover_w1")
IDS = [
    "i2ekx739xujtxkdf3y8br", "i8knk0nn798wzgjfyausm", "i3pli4pqni7w1ofihlz3o",
    "i8fl6q4uxuxfp787l4e09", "i36ukose2hd235akuu0vk", "isem6ra6afmg8104jqrum",
    "icq39c3avp3zfps0ikt1j", "i0hgdurh2k0on0xvsrl9g", "imw2thvdrsxwgptqamwxa",
    "i7e2hozw2v7asrsxnu1h4",
    "i7sjf4er5tklbvg5zwwyx", "il502f19y8sbkbhwjcntd", "icajijx5qgwafjkuo07v3",
    "izj3og43m0gj6mlc3s2m5", "ionicuaq0joiixtfmbndp",
]
PULL_SEM = threading.Semaphore(4)
LOCK = threading.Lock()
RESULTS = {}


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("E2B_KEY not found in .env")


def pull_with_retry(sb, path, tag, tries=3):
    for k in range(tries):
        try:
            with PULL_SEM:
                return sb.files.read(path, format="bytes")
        except Exception as e:
            with LOCK:
                print(f"[{tag}] pull retry {k + 1}/{tries}: {repr(e)[:100]}",
                      flush=True)
            time.sleep(45)
    raise RuntimeError("pull failed after retries")


def recover(i, sid):
    tag = f"rec{i:02d}"
    rec = {"sandbox": sid}
    sb = None
    try:
        from e2b import Sandbox
        sb = Sandbox.connect(sandbox_id=sid, timeout=3600)
        try:
            sb.set_timeout(3600)
        except Exception:
            pass
        with LOCK:
            print(f"[{tag}] connected {sid}", flush=True)
        # wait until run_shard fully exits ([r] trick: pgrep must not match
        # its own wrapping shell's cmdline)
        for _ in range(60):
            r = sb.commands.run(
                "pgrep -f '[r]un_shard.py' >/dev/null && echo 1 || echo 0",
                timeout=60)
            if r.stdout.strip() == "0":
                break
            time.sleep(30)
        rec["shard_done"] = True
        t0 = time.monotonic()
        r = sb.commands.run("cd /home/user/opion && rm -f shard.tgz && "
                            "tar -czf shard.tgz data", timeout=1200)
        rec["tar_rc"] = r.exit_code
        if r.exit_code != 0:
            raise RuntimeError(f"tar rc={r.exit_code}: {r.stderr[:120]}")
        data = pull_with_retry(sb, "/home/user/opion/shard.tgz", tag)
        rec["pull_mb"] = round(len(data) / 1e6, 1)
        rec["pull_s"] = round(time.monotonic() - t0, 1)
        assert data[:2] == b"\x1f\x8b", "bad gzip magic"
        os.makedirs(os.path.join(OUT, tag), exist_ok=True)
        with open(os.path.join(OUT, tag, "shard.tgz"), "wb") as f:
            f.write(data)
        try:
            info = sb.files.read("/home/user/opion/data/summary.json",
                                 format="bytes")
            with open(os.path.join(OUT, tag, "summary.json"), "wb") as f:
                f.write(info)
            rec["summary"] = json.loads(info.decode())
        except Exception as e:
            rec["summary_err"] = repr(e)[:120]
        rec["pulled"] = True
        sb.beta_pause()  # PAUSE — never kill
        rec["paused"] = True
        s = rec.get("summary", {}) or {}
        with LOCK:
            print(f"[{tag}] PAUSED teacher={s.get('teacher_decisions', '?')} "
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
    RESULTS[i] = rec


def main():
    os.environ["E2B_API_KEY"] = load_key()
    print(f"[recover] {len(IDS)} sandboxes", flush=True)
    t0 = time.monotonic()
    ths = []
    for i, sid in enumerate(IDS):
        th = threading.Thread(target=recover, args=(i, sid), daemon=True)
        th.start()
        ths.append(th)
        time.sleep(0.4)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            n = sum(1 for r in RESULTS.values())
        print(f"[recover] t={(time.monotonic() - t0) / 60:.1f}min "
              f"finished={n}/{len(IDS)}", flush=True)
    for th in ths:
        th.join()
    ok = [r for r in RESULTS.values() if r.get("pulled")]
    tot = sum((r.get("summary") or {}).get("teacher_decisions", 0)
              for r in RESULTS.values())
    errs = [r["sandbox"] for r in RESULTS.values() if "error" in r]
    with open(os.path.join(OUT, "recover_summary.json"), "w") as f:
        json.dump({"pulled": len(ok), "total_teacher": tot, "errors": errs,
                   "sandboxes": [RESULTS[i] for i in sorted(RESULTS)]},
                  f, indent=1, default=str)
    print(f"RECOVER_DONE pulled={len(ok)}/{len(IDS)} total_teacher={tot} "
          f"errors={errs} wall={(time.monotonic() - t0) / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
