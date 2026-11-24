#!/usr/bin/env python3
"""v2 live check on the test sandbox: 8 matches (4 SELF + 4 vs teacher) with
the re-distilled weights; verify bucket mix + rewards on the sandbox; pause."""
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


MIX = [{"opp": "SELF", "count": 4},
       {"opp": "bot/elite_instr.py", "count": 4}]

VERIFY = r"""
import glob, json
import numpy as np
import np_policy, sys
sys.path.insert(0, "/home/user/opion/rl")
sd = np_policy.load_pt("/home/user/opion/rl/current/final_act_v2.pt")["model"]
print("v2 params:", f"{int(sum(v.size for v in sd.values())):,}")
for f in sorted(glob.glob("/home/user/opion/rl/iter_v2/m*/manifest.json")):
    m = json.load(open(f))
    print(f"  {m['match']} {m['opp']:<22} rc={m['rc']} dec={m['decisions']}"
          f" hands={m['hands']} miss={m['dec_hands_missing']} valid={m['valid']}")
for tag, pref in (("SELF", "m000"), ("TEACHER", "m004")):
    bc = np.zeros(6)
    n = 0
    for f in sorted(glob.glob(f"/home/user/opion/rl/iter_v2/{pref}*/rollout.npz")):
        z = np.load(f)
        bc += np.bincount(z["bucket"], minlength=6)
        n += len(z["bucket"])
    if n:
        print(f"{tag} bucket mix (n={n:,})", np.round(bc / n, 3).tolist(),
              " teacher ref [0.17, 0.44, 0.10, 0.26, 0.04, 0.00]")
res = {}
for f in sorted(glob.glob("/home/user/opion/rl/iter_v2/m*/rollout.npz")):
    m = json.load(open(f.replace("rollout.npz", "manifest.json")))
    z = np.load(f)
    r = z["reward"].astype(np.float64)
    res.setdefault(m["opp"], []).append((m["match"], float(r.mean()),
                                         float(r.std()), len(r)))
for opp, rows in res.items():
    w = sum(x[3] for x in rows)
    mean = sum(x[1] * x[3] for x in rows) / w
    print(f"{opp}: mean reward/hand (student seat) = {mean:+.4f} "
          f"over {w} hands")
print("VERIFY_DONE")
"""


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox

    ids = json.load(open("/tmp/census_test_ids.json"))
    sb = Sandbox.connect(ids["test"], timeout=300)
    sb.set_timeout(3600)
    print(f"[v2] connected {sb.sandbox_id}", flush=True)

    rl = "/home/user/opion/rl"
    cur = os.path.join(rl, "current")
    with open(os.path.join(ROOT, "opion", "train", "final_act_v2.pt"), "rb") as f:
        sb.files.write(os.path.join(cur, "final_act_v2.pt"), f.read())
    sb.files.write(os.path.join(rl, "mix8.json"), json.dumps(MIX).encode())
    print("[v2] uploaded final_act_v2.pt", flush=True)

    t0 = time.time()
    r = sb.commands.run(
        f"cd {rl} && python3 rl_play.py --mix {rl}/mix8.json "
        f"--out {rl}/iter_v2 --weights {cur}/final_act_v2.pt "
        f"--disc {cur}/final_disc.pt --sb t2 --workers 8 --timeout-s 600",
        timeout=1200)
    print(r.stdout[-900:], flush=True)
    if r.stderr:
        print("[v2] stderr tail:", r.stderr[-400:], flush=True)
    print(f"[v2] wall={time.time()-t0:.0f}s", flush=True)

    v = sb.commands.run(f"cd {rl} && python3 - << 'EOF'\n{VERIFY}\nEOF",
                        timeout=300)
    print(v.stdout, flush=True)

    try:
        sb.beta_pause()
        print("[v2] sandbox paused", flush=True)
    except Exception as e:
        print(f"[v2] pause note: {e!r}", flush=True)
    print("V2_LIVE_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
