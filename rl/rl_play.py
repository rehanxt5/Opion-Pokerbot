#!/usr/bin/env python3
"""RL rollout runner (one sandbox): play the iteration's match mix with the
student (numpy port) inside the REAL engine, 8 concurrent matches on 8 vCPU.

python3 rl_play.py --mix mix.json --out /home/user/opion/rl/iter_0001 \
    --weights /home/user/opion/rl/current/final_act.pt \
    --disc /home/user/opion/rl/current/final_disc.pt \
    [--sb t0] [--workers 8] [--hands 400] [--timeout-s 900]

mix.json: [{"opp": "SELF" | "opponents/elite83.py" | "bot/elite_instr.py"
            | "opponents/styles/station.py" | ...}, ...]  (paths relative to
/home/user/opion; SELF = student vs student, both seats are rollouts)

Per match (8 workers, one engine each):
  - seed = rl:{sb}:{i}          (§6.6 rl: namespace, unique per match)
  - engine subprocess logs obs/meta under <out>/m{i}/
  - the student writes a rollout sidecar (choices + logp + per-hand nets)
  - after the engine exits the worker joins rewards, hard-checks the match
    (rc==0, every decision hand has a net, no empty matches) and saves
    rollout.npz + manifest.json under <out>/m{i}/ — NOTHING is downloaded.
  - one rerun on failure (same seed, §6.6 rerun rule)

rollout.npz arrays (student decisions only; self-mirror includes both pids):
  tokens[N,41]u8 ctx[N,17]f16 legal[N,6]u8 bucket[N]u8 logp[N]f32
  pid[N]u32 seat[N]u8 hand[N]u16 seq[N]u16 street[N]u8
  reward[N]f16 (per-hand net/200, repeated across the hand's decisions)
  amt[N]u16 pot[N]u16   (for offline ctx[14] re-encoding variants)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OPION = "/home/user/opion"
ENGINE = os.path.join(OPION, "engine_log.py")
STUDENT = os.path.join(HERE, "student_bot.py")

ACT_KEYS = ("tokens", "ctx", "legal", "bucket", "logp", "pid", "seat",
            "hand", "seq", "street", "reward", "amt", "pot")


def label(path):
    return os.path.splitext(os.path.basename(path))[0]


def play_one(job):
    i, opp, seed, a = job
    mdir = os.path.join(a.out, f"m{i:04d}")
    ld = os.path.join(mdir, "logs")
    os.makedirs(ld, exist_ok=True)
    obs = os.path.join(mdir, "obs.jsonl")
    meta = os.path.join(mdir, "meta.jsonl")
    sidecar = os.path.join(mdir, "rollout.jsonl")

    def run_once():
        if os.path.exists(sidecar):
            os.remove(sidecar)
        bot_b = STUDENT if opp == "SELF" else os.path.join(OPION, opp)
        bot_a = STUDENT  # student is always seat A here; mirror legs balance
        name_b = "self" if opp == "SELF" else label(opp)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPION_ACT_PT=a.weights,
                   OPION_DISC_PT=a.disc, OPION_ROLLOUT_LOG=sidecar)
        if a.weights_npz:
            env["OPION_ACT_NPZ"] = a.weights_npz
        cmd = [sys.executable, ENGINE, bot_a, bot_b,
               "--stage", "1", "--hands", str(a.hands), "--seed", seed,
               "--log-dir", ld, "--obs-log", obs, "--meta-log", meta,
               "--name-a", "opion_student", "--name-b", name_b]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=OPION, env=env,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               timeout=a.timeout_s)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        return rc, time.time() - t0

    rc, wall = run_once()
    res = build_rollout(mdir, sidecar, rc, wall, i, opp, seed, a.hands)
    if not res["valid"]:  # one rerun, same seed (§6.6)
        rc, wall = run_once()
        res = build_rollout(mdir, sidecar, rc, wall, i, opp, seed, a.hands)
        res["rerun"] = True
    for f in (sidecar,):
        if os.path.exists(f):
            os.remove(f)
    return res


def build_rollout(mdir, sidecar, rc, wall, i, opp, seed, hands):
    decs, rewards = [], {}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("r"):
                    rewards[(r["pid"], r["hand"])] = r["net"]
                elif "tokens" in r:
                    decs.append(r)
    by_pid_hand = {}
    for d in decs:
        by_pid_hand.setdefault((d["pid"], d["hand"]), []).append(d)
    missing = sum(1 for k in by_pid_hand if k not in rewards)
    cols = {k: [] for k in ACT_KEYS}
    for (pid, hand), lst in sorted(by_pid_hand.items()):
        net = rewards.get((pid, hand))
        if net is None:
            continue
        rew = net / 200.0
        for d in sorted(lst, key=lambda x: x["seq"]):
            cols["tokens"].append(d["tokens"])
            cols["ctx"].append(d["ctx"])
            cols["legal"].append(d["legal"])
            cols["bucket"].append(d["b"])
            cols["logp"].append(d["logp"])
            cols["pid"].append(pid & 0xFFFFFFFF)
            cols["seat"].append(d["seat"])
            cols["hand"].append(hand & 0xFFFF)
            cols["seq"].append(d["seq"] & 0xFFFF)
            cols["street"].append(d["st"])
            cols["reward"].append(rew)
            cols["amt"].append(d["amt"] & 0xFFFF)
            cols["pot"].append(d["pot"] & 0xFFFF)
    valid = (rc == 0 and len(rewards) >= hands - 2 and missing == 0
             and len(cols["bucket"]) > 0)
    res = {"match": f"m{i:04d}", "opp": opp, "seed": seed, "rc": rc,
           "wall_s": round(wall, 1), "decisions": len(cols["bucket"]),
           "hands": len(rewards), "dec_hands_missing": missing,
           "reward_mean_hand": round(sum(rewards.values())
                                     / max(1, len(rewards)) / 200.0, 4),
           "pids": sorted({p for (p, _h) in by_pid_hand}),
           "valid": valid}
    if cols["bucket"]:
        import numpy as np
        arrays = {
            "tokens": np.array(cols["tokens"], dtype=np.uint8),
            "ctx": np.array(cols["ctx"], dtype=np.float16),
            "legal": np.array(cols["legal"], dtype=np.uint8),
            "bucket": np.array(cols["bucket"], dtype=np.uint8),
            "logp": np.array(cols["logp"], dtype=np.float32),
            "pid": np.array(cols["pid"], dtype=np.uint32),
            "seat": np.array(cols["seat"], dtype=np.uint8),
            "hand": np.array(cols["hand"], dtype=np.uint16),
            "seq": np.array(cols["seq"], dtype=np.uint16),
            "street": np.array(cols["street"], dtype=np.uint8),
            "reward": np.array(cols["reward"], dtype=np.float16),
            "amt": np.array(cols["amt"], dtype=np.uint16),
            "pot": np.array(cols["pot"], dtype=np.uint16),
        }
        np.savez_compressed(os.path.join(mdir, "rollout.npz"), **arrays)
    with open(os.path.join(mdir, "manifest.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-npz", default="",
                    help="preferred: npz weights (set OPION_ACT_NPZ)")
    ap.add_argument("--disc", required=True)
    ap.add_argument("--sb", default="t0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--timeout-s", type=int, default=1200)
    a = ap.parse_args()

    mix = json.load(open(a.mix))
    jobs = []
    i = 0
    for entry in mix:
        for _ in range(int(entry["count"])):
            jobs.append((i, entry["opp"], f"rl:{a.sb}:{i}", a))
            i += 1
    os.makedirs(a.out, exist_ok=True)
    print(f"[rlplay] {len(jobs)} matches, {a.workers} workers -> {a.out}",
          flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(play_one, jobs, chunksize=1)
    total_dec = sum(r["decisions"] for r in results)
    invalid = [r["match"] for r in results if not r["valid"]]
    out = {"matches": len(jobs), "valid": len(jobs) - len(invalid),
           "invalid": invalid, "decisions": total_dec,
           "wall_s": round(time.time() - t0, 1),
           "per_opp": {}}
    for r in results:
        o = out["per_opp"].setdefault(r["opp"], {"matches": 0, "decisions": 0,
                                                 "wall_s": 0.0})
        o["matches"] += 1
        o["decisions"] += r["decisions"]
        o["wall_s"] += r["wall_s"]
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    print("RLPLAY_OK" if not invalid else "RLPLAY_INVALID", flush=True)


if __name__ == "__main__":
    main()
