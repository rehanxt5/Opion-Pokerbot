#!/usr/bin/env python3
"""Speed + sanity probe for the two external submission zips:
1 match vs A (elite next-gen v8, single-file neural+CFR) and 1 vs
B (synergy v8.8 package), on the test sandbox with champion weights.
Measures per-match wall time so the full screen can be sized."""
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


def main():
    os.environ["E2B_API_KEY"] = load_key()
    from e2b import Sandbox
    ids = json.load(open("/tmp/census_test_ids.json"))
    sb = Sandbox.connect(ids["test"], timeout=300)
    sb.set_timeout(3600)
    print(f"[probe] connected {sb.sandbox_id}", flush=True)

    rl = "/home/user/opion/rl"
    sb.commands.run("mkdir -p /home/user/opion/opponents/ext/sub_a "
                    "/home/user/opion/opponents/ext/sub_b/data", timeout=60)
    with open("/tmp/opp_inspect/a/bot.py", "rb") as f:
        sb.files.write("/home/user/opion/opponents/ext/sub_a/bot.py", f.read())
    for fn in ("bot.py", "mcbase.py", "pbcore.py", "synergy.py",
               "delaa_dump.py", "tns.py", "synergy_probe.py", "nnu.py",
               "tmr.py", "exbr.py"):
        with open(f"/tmp/opp_inspect/b/{fn}", "rb") as f:
            sb.files.write(f"/home/user/opion/opponents/ext/sub_b/{fn}",
                           f.read())
    for fn in ("tns_w.npz", "nnu_w.npz", "preflop_equity.npz"):
        with open(f"/tmp/opp_inspect/b/data/{fn}", "rb") as f:
            sb.files.write(f"/home/user/opion/opponents/ext/sub_b/data/{fn}",
                           f.read())
    print("[probe] uploaded A + B", flush=True)

    mix = [{"opp": "opponents/ext/sub_a/bot.py", "count": 1},
           {"opp": "opponents/ext/sub_b/bot.py", "count": 1}]
    sb.files.write(f"{rl}/mix_ext.json", json.dumps(mix).encode())
    t0 = time.time()
    r = sb.commands.run(
        f"cd {rl} && python3 rl_play.py --mix {rl}/mix_ext.json "
        f"--out {rl}/iter_ext_probe "
        f"--weights /home/user/opion/rl/current/final_act.pt "
        f"--weights-npz /home/user/opion/rl/current/policy_np.npz "
        f"--disc /home/user/opion/rl/current/final_disc.pt --sb ext "
        f"--workers 2 --timeout-s 900", timeout=2400)
    print(r.stdout[-1200:], flush=True)
    if r.stderr:
        print("stderr tail:", r.stderr[-400:], flush=True)
    print(f"[probe] wall: {time.time() - t0:.0f}s", flush=True)
    for f in (f"{rl}/iter_ext_probe/m0000/manifest.json",
              f"{rl}/iter_ext_probe/m0001/manifest.json"):
        try:
            m = json.loads(sb.files.read(f))
            print(f"  {m['opp']}: rc={m['rc']} wall={m['wall_s']}s "
                  f"hands={m['hands']} rew/hand={m['reward_mean_hand']:+.4f} "
                  f"valid={m['valid']}", flush=True)
        except Exception as e:
            print("  manifest read fail:", f, repr(e), flush=True)
    try:
        sb.beta_pause()
        print("[probe] sandbox paused", flush=True)
    except Exception as e:
        print(f"[probe] pause note: {e!r}", flush=True)
    print("EXT_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
