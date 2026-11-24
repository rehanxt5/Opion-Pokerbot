#!/usr/bin/env python3
"""Dummy RL rollout test on the census-test sandbox (M1 exit test).

Uploads the RL kit (np_policy, spec, student_bot, rl_play + both .pt
weights), plays a 16-match mix with 8 workers, verifies the rollouts ON the
sandbox (manifest + npz stats printed via remote stdout — nothing is
downloaded), then pauses the sandbox (never kill).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load_key():
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


MIX = [
    {"opp": "SELF", "count": 6},
    {"opp": "bot/elite_instr.py", "count": 4},
    {"opp": "opponents/styles/station.py", "count": 3},
    {"opp": "opponents/styles/randombot.py", "count": 3},
]

VERIFY = r"""
import glob, json
import numpy as np
man = json.load(open("/home/user/opion/rl/iter_test/manifest.json"))
print("MANIFEST", json.dumps({k: man[k] for k in
      ("matches", "valid", "invalid", "decisions", "wall_s")}))
for f in sorted(glob.glob("/home/user/opion/rl/iter_test/m*/manifest.json")):
    m = json.load(open(f))
    print(f"  {m['match']} {m['opp']:<28} rc={m['rc']} dec={m['decisions']}"
          f" hands={m['hands']} miss={m['dec_hands_missing']}"
          f" wall={m['wall_s']}s valid={m['valid']}")
z = np.load("/home/user/opion/rl/iter_test/m0000/rollout.npz")
n = len(z["bucket"])
bc = np.bincount(z["bucket"], minlength=6) / n
r = z["reward"].astype(np.float64)
uniq_hand = len(set(zip(z["pid"].tolist(), z["hand"].tolist())))
print(f"DEEP m0000: n={n} pids={sorted(set(z['pid'].tolist()))}"
      f" (pid,hand) pairs={uniq_hand}")
print(f"  bucket mix {np.round(bc, 3).tolist()}  (teacher ref"
      f" [0.17, 0.44, 0.10, 0.26, 0.04, 0.00])")
print(f"  reward/hand mean={r.mean():+.3f} sd={r.std():.3f}"
      f" (per-hand net/200; self-mirror is zero-sum-ish)")
lp = z["logp"]
print(f"  logp range [{lp.min():.3f}, {lp.max():.3f}]"
      f" (sampled policy -> all <= 0)")
agg = [json.load(open(f)) for f in
       glob.glob("/home/user/opion/rl/iter_test/m*/manifest.json")]
tot = sum(m["decisions"] for m in agg)
hands = sum(m["hands"] for m in agg)
import os
size = sum(os.path.getsize(f) for f in
           glob.glob("/home/user/opion/rl/iter_test/m*/rollout.npz"))
print(f"TOTAL decisions={tot:,} hands={hands:,} rollout.npz bytes={size:,}")
print("VERIFY_DONE")
"""


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox

    ids = json.load(open("/tmp/census_test_ids.json"))
    sb = Sandbox.connect(ids["test"], timeout=300)  # auto-resume
    sb.set_timeout(3600)
    print(f"[test] connected {sb.sandbox_id}", flush=True)

    rl = "/home/user/opion/rl"
    cur = os.path.join(rl, "current")
    sb.commands.run(f"mkdir -p {cur}")
    FLEET = os.path.join(ROOT, "opion", "fleet")
    TRAIN = os.path.join(ROOT, "opion", "train")
    ups = [(os.path.join(HERE, "np_policy.py"), os.path.join(rl, "np_policy.py")),
           (os.path.join(FLEET, "spec.py"), os.path.join(rl, "spec.py")),
           (os.path.join(HERE, "student_bot.py"), os.path.join(rl, "student_bot.py")),
           (os.path.join(HERE, "rl_play.py"), os.path.join(rl, "rl_play.py")),
           (os.path.join(TRAIN, "final_act.pt"), os.path.join(cur, "final_act.pt")),
           (os.path.join(TRAIN, "final_disc.pt"), os.path.join(cur, "final_disc.pt"))]
    for local, remote in ups:
        with open(local, "rb") as f:
            data = f.read()
        sb.files.write(remote, data)
        print(f"[test] uploaded {os.path.basename(local)} -> {remote} "
              f"({len(data)/1e6:.1f}MB)", flush=True)
    sb.files.write(os.path.join(rl, "mix.json"),
                   json.dumps(MIX).encode())

    t0 = time.time()
    r = sb.commands.run(
        f"cd {rl} && python3 rl_play.py --mix {rl}/mix.json "
        f"--out {rl}/iter_test --weights {cur}/final_act.pt "
        f"--disc {cur}/final_disc.pt --sb t0 --workers 8 "
        f"--timeout-s 600", timeout=1200)
    print(r.stdout[-3000:], flush=True)
    if r.stderr:
        print("[test] stderr tail:", r.stderr[-600:], flush=True)
    print(f"[test] rl_play exit={r.exit_code} wall={time.time()-t0:.0f}s",
          flush=True)

    v = sb.commands.run(f"cd {rl} && python3 - << 'EOF'\n{VERIFY}\nEOF",
                        timeout=300)
    print(v.stdout, flush=True)
    if v.stderr:
        print("[verify] stderr:", v.stderr[-500:], flush=True)

    try:
        sb.beta_pause()
        print("[test] sandbox paused (all rollout data left on its disk)",
              flush=True)
    except Exception as e:
        print(f"[test] pause note: {e!r}", flush=True)
    print("RL_DUMMY_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
