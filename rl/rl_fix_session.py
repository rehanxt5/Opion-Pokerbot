#!/usr/bin/env python3
"""One sandbox session: run the amt diagnosis, upload the patched student
bot, rerun a small live mix (4 SELF + 4 teacher), verify bucket mixes on the
sandbox, pause. Nothing is downloaded."""
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


MIX8 = [{"opp": "SELF", "count": 4},
        {"opp": "bot/elite_instr.py", "count": 4}]

VERIFY = r"""
import glob, json
import numpy as np
tot = {}
for f in sorted(glob.glob("/home/user/opion/rl/iter_fix/m*/manifest.json")):
    m = json.load(open(f))
    print(f"  {m['match']} {m['opp']:<22} rc={m['rc']} dec={m['decisions']}"
          f" hands={m['hands']} miss={m['dec_hands_missing']} valid={m['valid']}")
for tag, pref in (("SELF", "m000"), ("TEACHER", "m004")):
    bc = np.zeros(6)
    n = 0
    for f in sorted(glob.glob(f"/home/user/opion/rl/iter_fix/{pref}*/rollout.npz")):
        z = np.load(f)
        bc += np.bincount(z["bucket"], minlength=6)
        n += len(z["bucket"])
    if n:
        print(f"{tag} bucket mix (n={n:,})", np.round(bc / n, 3).tolist(),
              " teacher ref [0.17, 0.44, 0.10, 0.26, 0.04, 0.00]")
r = np.concatenate([np.load(f)["reward"].astype(np.float64)
                    for f in glob.glob("/home/user/opion/rl/iter_fix/m*/rollout.npz")])
print(f"reward/hand mean={r.mean():+.4f} sd={r.std():.3f}")
print("VERIFY_DONE")
"""


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox

    ids = json.load(open("/tmp/census_test_ids.json"))
    sb = Sandbox.connect(ids["test"], timeout=300)
    sb.set_timeout(3600)
    print(f"[fix] connected {sb.sandbox_id}", flush=True)

    rl = "/home/user/opion/rl"
    cur = os.path.join(rl, "current")
    with open(os.path.join(HERE, "exp_amt.py"), "rb") as f:
        sb.files.write(os.path.join(rl, "exp_amt.py"), f.read())
    with open(os.path.join(HERE, "student_bot.py"), "rb") as f:
        sb.files.write(os.path.join(rl, "student_bot.py"), f.read())
    print("[fix] uploaded exp_amt.py + patched student_bot.py", flush=True)

    print("[fix] === amt-convention experiment ===", flush=True)
    r = sb.commands.run(f"cd {rl} && OMP_NUM_THREADS=4 python3 exp_amt.py",
                        timeout=900)
    print(r.stdout, flush=True)
    if r.stderr:
        print("[exp] stderr tail:", r.stderr[-500:], flush=True)
    confirmed = "CONFIRMED_FIX" in r.stdout

    print(f"[fix] === live rerun (patched bot, 8 matches) ===", flush=True)
    sb.files.write(os.path.join(rl, "mix8.json"), json.dumps(MIX8).encode())
    r = sb.commands.run(
        f"cd {rl} && python3 rl_play.py --mix {rl}/mix8.json "
        f"--out {rl}/iter_fix --weights {cur}/final_act.pt "
        f"--disc {cur}/final_disc.pt --sb t1 --workers 8 --timeout-s 600",
        timeout=1200)
    print(r.stdout[-1200:], flush=True)
    if r.stderr:
        print("[rlplay] stderr tail:", r.stderr[-400:], flush=True)

    v = sb.commands.run(f"cd {rl} && python3 - << 'EOF'\n{VERIFY}\nEOF",
                        timeout=300)
    print(v.stdout, flush=True)

    try:
        sb.beta_pause()
        print("[fix] sandbox paused", flush=True)
    except Exception as e:
        print(f"[fix] pause note: {e!r}", flush=True)
    print("RL_FIX_SESSION_DONE confirmed=" + str(confirmed), flush=True)


if __name__ == "__main__":
    main()
