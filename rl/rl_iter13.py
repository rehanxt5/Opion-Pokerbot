#!/usr/bin/env python3
"""RL iteration 5 rollouts: the same 8 sandboxes, same §9.3 mix (260
matches), played by the ITERATION-12 policy (policy_np_iter12.npz).
Serves as (a) PPO fuel for a potential iteration 13 and (b) the iteration-12
mini-screen (fresh seeds rl:i13:*, mirror legs) that places iter-12 in the
blind-final candidate set. Same hardened bundling as iter-3/4:
tag-prefixed shard names + in-bundle pid rewrite (pid + sb*2^22).
All sandboxes re-paused after. 20-concurrent cap respected (8 used).
"""
import glob
import json
import os
import shutil
import sys
import tarfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONC = 8
NSB = 8
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


def opp_of_global():
    out = {}
    i = 0
    for opp, count in MIX:
        for _ in range(count):
            out[i] = opp
            i += 1
    return out


def one(k, sb_id, chunk):
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
                (os.path.join(ROOT, "opion/rl/ckpts/policy_np_iter12.npz"),
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
            f"--out {rl}/iter_0013 --weights {cur}/final_act.pt "
            f"--weights-npz {cur}/policy_np.npz "
            f"--disc {cur}/final_disc.pt --sb i13.{k} --workers 8 "
            f"--timeout-s 600", timeout=2400)
        rec["exit"] = r.exit_code
        tail = r.stdout[-600:]
        rec["tail"] = tail
        if "RLPLAY_INVALID" in r.stdout or r.exit_code != 0:
            raise RuntimeError(f"rl_play failed: {tail[-200:]}")
        man = json.loads(sb.files.read(f"{rl}/iter_0013/manifest.json"))
        rec["manifest"] = {k2: man[k2] for k2 in
                           ("matches", "valid", "invalid", "decisions",
                            "wall_s")}
        # pull rollout npz + per-match manifests (fuel + screen data)
        dst = os.path.join(ROOT, "opion/data/rl/iter_0013", tag)
        os.makedirs(dst, exist_ok=True)
        pulled = 0
        for pattern, suffix in ((f"{rl}/iter_0013/m*/rollout.npz", ".npz"),
                                (f"{rl}/iter_0013/m*/manifest.json",
                                 ".man.json")):
            listing = sb.commands.run(f"ls {pattern}",
                                      timeout=120).stdout.split()
            for f in listing:
                base = os.path.basename(os.path.dirname(f))
                data = sb.files.read(f, format="bytes")
                with open(os.path.join(dst, base + suffix), "wb") as g:
                    g.write(data)
                if suffix == ".npz":
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
        print(f"[i13] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"dec={rec.get('manifest', {}).get('decisions', '?')} "
              f"pulled={rec.get('pulled', '?')} paused={rec.get('paused')} "
              f"({ok}/{len(RESULTS)} done)\n{rec.get('tail', '')[-300:]}",
              flush=True)


def screen(src, nsb):
    """Per-opponent mini-screen from pulled per-match manifests."""
    opp_map = opp_of_global()
    per = {}
    for k in range(nsb):
        gidx = sorted(g for g in opp_map if g % nsb == k)
        for pos, g in enumerate(gidx):
            mf = os.path.join(src, f"sb{k}", f"m{pos:04d}.man.json")
            if not os.path.exists(mf):
                continue
            m = json.load(open(mf))
            per.setdefault(opp_map[g], []).append(
                (m["reward_mean_hand"], m["hands"]))
    print("\n=== ITER-12 POLICY SCREEN (fresh deals rl:i13:*, mirror legs, "
          "stacks/hand from student seat) ===", flush=True)
    order = ["bot/elite_instr.py", "opponents/gen86_pkg/package_g18_6/bot.py",
             "opponents/elite83.py", "opponents/styles/maniac.py",
             "opponents/styles/station.py", "opponents/styles/rock.py",
             "opponents/styles/randombot.py", "opponents/styles/mixer.py",
             "SELF"]
    for opp in order:
        rows = per.get(opp)
        if not rows:
            continue
        n = sum(r[1] for r in rows)
        mean = sum(r[0] * r[1] for r in rows) / n
        if len(rows) > 1:
            var = sum(r[1] * (r[0] - mean) ** 2 for r in rows) / n
            se = (var / len(rows)) ** 0.5
        else:
            se = float("nan")
        print(f"  {opp:<45} {mean:+.4f} ± {se:.4f}  "
              f"({len(rows)} matches, {n:,} hands)", flush=True)
    print("=== SCREEN_DONE | iter-11 ref: elitev6 -0.0029, gen86 +0.0002, "
          "elite83 -0.0025 ===", flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    summ = json.load(open(os.path.join(ROOT,
                                       "opion/data/rl/iter_0002_summary.json")))
    sandbox_ids = [r["sandbox"] for r in summ["sandboxes"]]
    assert len(sandbox_ids) == NSB, sandbox_ids
    chunks = [[] for _ in range(NSB)]
    i = 0
    for opp, count in MIX:
        for _ in range(count):
            chunks[i % NSB].append((i, opp, f"rl:i13:{i}"))
            i += 1
    assert i == 260
    for c in chunks:
        c.sort()
    t0 = time.monotonic()
    sem = threading.Semaphore(CONC)

    def worker(k, sid, chunk):
        with sem:
            one(k, sid, chunk)

    ths = []
    for k, sid in enumerate(sandbox_ids):
        th = threading.Thread(target=worker, args=(k, sid, chunks[k]),
                              daemon=True)
        th.start()
        ths.append(th)
        time.sleep(2)
    print(f"[i13] {NSB} sandboxes, 260 matches total", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(120)
        with LOCK:
            print(f"[i5 monitor] t={(time.monotonic()-t0)/60:.1f}min "
                  f"done={len(RESULTS)}/8", flush=True)
    for th in ths:
        th.join()

    out = {"iter": 13, "sandboxes": [RESULTS[k] for k in sorted(RESULTS)]}
    with open(os.path.join(ROOT, "opion/data/rl/iter_0013_summary.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    tot_dec = sum(r.get("manifest", {}).get("decisions", 0)
                  for r in RESULTS.values())
    errs = [r["tag"] for r in RESULTS.values() if not r.get("ok")]
    print(f"[i13] total decisions={tot_dec:,} errors={errs}", flush=True)

    # bundle a potential PPO iteration-13 tar (pid rewrite + tag-prefixed)
    src = os.path.join(ROOT, "opion/data/rl/iter_0013")
    tmp = "/tmp/rl_iter13"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "rollouts"), exist_ok=True)
    rows = 0
    for k in range(NSB):
        tag = f"sb{k}"
        for f in sorted(glob.glob(os.path.join(src, tag, "*.npz"))):
            z = dict(np.load(f, allow_pickle=False))
            p = z["pid"].astype(np.uint64) + np.uint64(k) * np.uint64(2 ** 22)
            z["pid"] = p.astype(np.uint32)
            rows += int(z["bucket"].size)
            np.savez_compressed(
                os.path.join(tmp, "rollouts", f"{tag}_{os.path.basename(f)}"),
                **z)
    with open(os.path.join(tmp, "beta.txt"), "w") as f:
        f.write("10.0\n")
    shutil.copy(os.path.join(HERE, "np_policy.py"),
                os.path.join(tmp, "np_policy.py"))
    shutil.copy(os.path.join(ROOT, "opion/rl/ckpts/policy_rl_iter12.pt"),
                os.path.join(tmp, "policy_rl.pt"))
    tar = "/tmp/rl_iter_0013.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(tmp, arcname="rl_iter")
    shutil.copy(tar, os.path.join(ROOT, "opion/data/rl/rl_iter_0013.tar"))
    print(f"[i13] tar: {tar} ({os.path.getsize(tar)/1e6:.1f}MB, "
          f"{rows:,} rows, pids sandbox-unique)", flush=True)

    screen(src, NSB)
    print("RL_ITER13_DONE", flush=True)


if __name__ == "__main__":
    main()
