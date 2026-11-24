#!/usr/bin/env python3
"""RL iteration 2 rollouts: 8 sandboxes (test + 7 wave-1) x ~33 matches each
= the full §9.3 production mix (260 matches, ~280k decisions), played by the
iteration-1 policy (policy_np_iter1.npz). Rollout npz files are pulled back
(small — they are the PPO fuel); everything else stays on the sandboxes.
All sandboxes re-paused after. 20-concurrent cap respected (8 used).
"""
import json
import os
import sys
import threading
import time
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONC = 8
RESULTS = {}
LOCK = threading.Lock()

MIX = [("SELF", 60), ("bot/elite_instr.py", 40),
       ("opponents/gen86_pkg/package_g18_6/bot.py", 30),
       ("opponents/elite83.py", 30), ("opponents/styles/maniac.py", 20),
       ("opponents/styles/station.py", 20), ("opponents/styles/rock.py", 20),
       ("opponents/styles/randombot.py", 20), ("opponents/styles/mixer.py", 20)]


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def build_jobs(nsb):
    jobs = []
    i = 0
    for opp, count in MIX:
        for _ in range(count):
            jobs.append((i, opp, f"rl:i2:{i}"))
            i += 1
    assert len(jobs) == 260
    chunks = [[] for _ in range(nsb)]
    for j in jobs:  # round-robin for balance
        chunks[j[0] % nsb].append(j)
    for c in chunks:  # renumber per-sandbox match dirs
        c.sort()
    return chunks


def one(k, sb_id, chunk, sb_args):
    tag = f"sb{k}"
    rec = {"tag": tag, "sandbox": sb_id, "matches": len(chunk)}
    from e2b import Sandbox
    sb = None
    try:
        last = None
        for attempt in range(4):
            try:
                sb = Sandbox.connect(sb_id, timeout=300)
                break
            except Exception as e:
                last = e
                time.sleep(30 * (attempt + 1))
        if sb is None:
            raise RuntimeError(f"connect failed: {last!r}")
        sb.set_timeout(5400)
        rl = "/home/user/opion/rl"
        cur = os.path.join(rl, "current")
        sb.commands.run(f"mkdir -p {cur}", timeout=60)
        for local, remote in (
                (os.path.join(HERE, "np_policy.py"), f"{rl}/np_policy.py"),
                (os.path.join(ROOT, "opion/fleet/spec.py"), f"{rl}/spec.py"),
                (os.path.join(HERE, "student_bot.py"), f"{rl}/student_bot.py"),
                (os.path.join(HERE, "rl_play.py"), f"{rl}/rl_play.py"),
                (os.path.join(ROOT, "opion/rl/ckpts/policy_np_iter1.npz"),
                 f"{cur}/policy_np.npz"),
                (os.path.join(ROOT, "opion/train/final_disc.pt"),
                 f"{cur}/final_disc.pt"),
                (os.path.join(ROOT, "opion/train/final_act.pt"),
                 f"{cur}/final_act.pt")):
            with open(local, "rb") as f:
                sb.files.write(remote, f.read())
        mix = [{"opp": opp, "count": 1} for (_i, opp, _s) in chunk]
        sb.files.write(f"{rl}/mix.json", json.dumps(mix).encode())
        r = sb.commands.run(
            f"cd {rl} && python3 rl_play.py --mix {rl}/mix.json "
            f"--out {rl}/iter_0002 --weights {cur}/final_act.pt "
            f"--weights-npz {cur}/policy_np.npz "
            f"--disc {cur}/final_disc.pt --sb i2.{k} --workers 8 "
            f"--timeout-s 600", timeout=2400)
        rec["exit"] = r.exit_code
        tail = r.stdout[-600:]
        rec["tail"] = tail
        if "RLPLAY_INVALID" in r.stdout or r.exit_code != 0:
            raise RuntimeError(f"rl_play failed: {tail[-200:]}")
        man = json.loads(sb.files.read(f"{rl}/iter_0002/manifest.json"))
        rec["manifest"] = {k2: man[k2] for k2 in
                           ("matches", "valid", "invalid", "decisions",
                            "wall_s")}
        # pull rollout npz + manifests (the PPO fuel)
        os.makedirs(os.path.join(ROOT, "opion/data/rl/iter_0002", tag),
                    exist_ok=True)
        pulled = 0
        listing = sb.commands.run(
            f"ls {rl}/iter_0002/m*/rollout.npz", timeout=120).stdout.split()
        for f in listing:
            base = os.path.basename(os.path.dirname(f))
            data = sb.files.read(f, format="bytes")
            with open(os.path.join(ROOT, "opion/data/rl/iter_0002", tag,
                                   base + ".npz"), "wb") as g:
                g.write(data)
            pulled += 1
        rec["pulled"] = pulled
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
        RESULTS[k] = rec
        ok = sum(1 for x in RESULTS.values() if x.get("ok"))
        print(f"[i2] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"dec={rec.get('manifest', {}).get('decisions', '?')} "
              f"pulled={rec.get('pulled', '?')} paused={rec.get('paused')} "
              f"({ok}/{len(RESULTS)} done)\n{rec.get('tail', '')[-300:]}",
              flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    w1 = json.load(open("/tmp/wave1_ids.json"))
    ids = json.load(open("/tmp/census_test_ids.json"))
    sandbox_ids = [ids["test"]] + w1[:7]
    chunks = build_jobs(len(sandbox_ids))
    t0 = time.monotonic()
    sem = threading.Semaphore(CONC)

    def worker(k, sid, chunk):
        with sem:
            one(k, sid, chunk, None)

    ths = []
    for k, sid in enumerate(sandbox_ids):
        th = threading.Thread(target=worker, args=(k, sid, chunks[k]),
                              daemon=True)
        th.start()
        ths.append(th)
        time.sleep(2)
    print(f"[i2] {len(sandbox_ids)} sandboxes, 260 matches total", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(120)
        with LOCK:
            print(f"[i2 monitor] t={(time.monotonic()-t0)/60:.1f}min "
                  f"done={len(RESULTS)}/8", flush=True)
    for th in ths:
        th.join()

    out = {"iter": 2, "sandboxes": [RESULTS[k] for k in sorted(RESULTS)]}
    with open(os.path.join(ROOT, "opion/data/rl/iter_0002_summary.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    tot_dec = sum(r.get("manifest", {}).get("decisions", 0)
                  for r in RESULTS.values())
    errs = [r["tag"] for r in RESULTS.values() if not r.get("ok")]
    print(f"[i2] total decisions={tot_dec:,} errors={errs}", flush=True)

    # bundle the PPO tar for iteration 2
    src = os.path.join(ROOT, "opion/data/rl/iter_0002")
    tmp = "/tmp/rl_iter"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "rollouts"), exist_ok=True)
    for dirpath, _d, files in os.walk(src):
        for fn in files:
            if fn.endswith(".npz"):
                shutil.copy(os.path.join(dirpath, fn),
                            os.path.join(tmp, "rollouts", fn))
    with open(os.path.join(tmp, "beta.txt"), "w") as f:
        f.write("10.0\n")
    shutil.copy(os.path.join(HERE, "np_policy.py"),
                os.path.join(tmp, "np_policy.py"))
    shutil.copy(os.path.join(ROOT, "opion/rl/ckpts/policy_rl_iter1.pt"),
                os.path.join(tmp, "policy_rl.pt"))
    tar = "/tmp/rl_iter_0002.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(tmp, arcname="rl_iter")
    print(f"[i2] tar: {tar} ({os.path.getsize(tar)/1e6:.1f}MB)", flush=True)
    print("RL_ITER2_DONE", flush=True)


if __name__ == "__main__":
    main()
