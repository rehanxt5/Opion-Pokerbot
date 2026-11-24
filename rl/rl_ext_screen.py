#!/usr/bin/env python3
"""External-submission screen: champion (iter-14) vs the two submission
zips the user received — A = elite next-gen v8 (single-file neural+CFR),
B = synergy v8.8 package. 24 matches each across the usual 8 sandboxes,
fresh seeds rl:ext:*, mirror legs. Manifests pulled; sandboxes re-paused."""
import glob
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CONC = 8
NSB = 8
RESULTS = {}
LOCK = threading.Lock()

MIX = [("opponents/ext/sub_a/bot.py", 24),
       ("opponents/ext/sub_b/bot.py", 24)]

B_FILES = ("bot.py", "mcbase.py", "pbcore.py", "synergy.py", "delaa_dump.py",
           "tns.py", "synergy_probe.py", "nnu.py", "tmr.py", "exbr.py")
B_DATA = ("tns_w.npz", "nnu_w.npz", "preflop_equity.npz")


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
        # upload both external bots
        sb.commands.run("mkdir -p /home/user/opion/opponents/ext/sub_a "
                        "/home/user/opion/opponents/ext/sub_b/data",
                        timeout=60)
        with open("/tmp/opp_inspect/a/bot.py", "rb") as f:
            sb.files.write("/home/user/opion/opponents/ext/sub_a/bot.py",
                           f.read())
        for fn in B_FILES:
            with open(f"/tmp/opp_inspect/b/{fn}", "rb") as f:
                sb.files.write(f"/home/user/opion/opponents/ext/sub_b/{fn}",
                               f.read())
        for fn in B_DATA:
            with open(f"/tmp/opp_inspect/b/data/{fn}", "rb") as f:
                sb.files.write(f"/home/user/opion/opponents/ext/sub_b/data/{fn}",
                               f.read())
        # champion weights
        for local, remote in (
                (os.path.join(ROOT, "opion/rl/ckpts/policy_np_iter14.npz"),
                 f"{cur}/policy_np.npz"),
                (os.path.join(HERE, "np_policy.py"), f"{rl}/np_policy.py"),
                (os.path.join(ROOT, "opion/fleet/spec.py"), f"{rl}/spec.py"),
                (os.path.join(HERE, "student_bot.py"), f"{rl}/student_bot.py"),
                (os.path.join(HERE, "rl_play.py"), f"{rl}/rl_play.py"),
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
            f"--out {rl}/iter_ext --weights {cur}/final_act.pt "
            f"--weights-npz {cur}/policy_np.npz "
            f"--disc {cur}/final_disc.pt --sb ext.{k} --workers 6 "
            f"--timeout-s 900", timeout=1800)
        rec["exit"] = r.exit_code
        rec["tail"] = r.stdout[-400:]
        if "RLPLAY_INVALID" in r.stdout or r.exit_code != 0:
            raise RuntimeError(f"rl_play failed: {r.stdout[-200:]}")
        dst = os.path.join(ROOT, "opion/data/rl/iter_ext", tag)
        os.makedirs(dst, exist_ok=True)
        listing = sb.commands.run(f"ls {rl}/iter_ext/m*/manifest.json",
                                  timeout=120).stdout.split()
        for f in listing:
            base = os.path.basename(os.path.dirname(f))
            data = sb.files.read(f, format="bytes")
            with open(os.path.join(dst, base + ".man.json"), "wb") as g:
                g.write(data)
        rec["pulled"] = len(listing)
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
        print(f"[ext] {tag}: {'OK' if rec.get('ok') else 'FAIL'} "
              f"pulled={rec.get('pulled', '?')} paused={rec.get('paused')} "
              f"({ok}/{len(RESULTS)} done) {rec.get('error', '')[:120]}",
              flush=True)


def main():
    os.environ["E2B_API_KEY"] = load_key()
    summ = json.load(open(os.path.join(ROOT,
                                       "opion/data/rl/iter_0002_summary.json")))
    sandbox_ids = [r["sandbox"] for r in summ["sandboxes"]]
    assert len(sandbox_ids) == NSB
    chunks = [[] for _ in range(NSB)]
    i = 0
    for opp, count in MIX:
        for _ in range(count):
            chunks[i % NSB].append((i, opp, f"rl:ext:{i}"))
            i += 1
    assert i == 48
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
    print(f"[ext] {NSB} sandboxes, {i} matches total", flush=True)
    while any(th.is_alive() for th in ths):
        time.sleep(60)
        with LOCK:
            print(f"[ext monitor] t={(time.monotonic()-t0)/60:.1f}min "
                  f"done={len(RESULTS)}/8", flush=True)
    for th in ths:
        th.join()

    errs = [r["tag"] for r in RESULTS.values() if not r.get("ok")]
    print(f"[ext] errors={errs}", flush=True)

    # screen from pulled manifests
    opp_map = opp_of_global()
    src = os.path.join(ROOT, "opion/data/rl/iter_ext")
    per = {}
    for k in range(NSB):
        gidx = sorted(g for g in opp_map if g % NSB == k)
        for pos, g in enumerate(gidx):
            mf = os.path.join(src, f"sb{k}", f"m{pos:04d}.man.json")
            if not os.path.exists(mf):
                continue
            m = json.load(open(mf))
            per.setdefault(opp_map[g], []).append(
                (m["reward_mean_hand"], m["hands"]))
    print("\n=== CHAMPION (iter-14) vs EXTERNAL SUBMISSIONS "
          "(fresh deals rl:ext:*, mirror legs, stacks/hand) ===", flush=True)
    names = {"opponents/ext/sub_a/bot.py": "A: elite next-gen v8",
             "opponents/ext/sub_b/bot.py": "B: synergy v8.8"}
    for opp in opp_map.values():
        rows = per.get(opp)
        if not rows:
            continue
        n = sum(r[1] for r in rows)
        mean = sum(r[0] * r[1] for r in rows) / n
        var = sum(r[1] * (r[0] - mean) ** 2 for r in rows) / n
        se = (var / len(rows)) ** 0.5
        print(f"  {names.get(opp, opp):<28} champion {mean:+.4f} ± {se:.4f} "
              f"({len(rows)} matches, {n:,} hands)", flush=True)
    print("EXT_SCREEN_DONE", flush=True)


if __name__ == "__main__":
    main()
