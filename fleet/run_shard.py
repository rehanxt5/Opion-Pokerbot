#!/usr/bin/env python3
"""opion dataset generation v2: round-looped, heterogeneous matches.

python3 run_shard.py --bot /home/user/opion/bot/elite_instr.py \
    --teacher-name elitev6 --opponents pairs.json --quota 1250000 \
    [--matches 8] [--hands 400] [--stage 1] [--wave 1] [--sb 0] \
    [--engine engine_log.py] [--out-dir /data] [--gzip] [--timeout-s 2700]

- Each ROUND launches --matches matches with fresh, globally-unique seeds
  ("w{wave}:{sb}:r{round}:m{i}"); the round runs to COMPLETION (no mid-write
  kills) and the loop stops at the first round boundary where the cumulative
  TEACHER decision count (is_t=true act+disc records) reaches --quota.
- Opponents are sampled per match from --opponents (JSON: [{"path": ..., "w": ...}]);
  teacher seat alternates A/B across matches for seat balance.
- Every obs record carries seed/bot/opp_bot/bpid/is_t/seq/req_echo (schema v2);
  obs/meta are separate streams; soft labels come from the teacher's own
  OPION_SOFT_LOG sidecar (joined at merge by pid+hand+seq).
"""
import argparse
import glob
import gzip
import json
import os
import random
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bot", required=True, help="teacher bot path")
    p.add_argument("--teacher-name", required=True)
    p.add_argument("--opponents", required=True,
                   help="JSON: [{\"path\": rel-or-abs, \"w\": weight}, ...]; "
                        "path may contain {SELF} for the teacher")
    p.add_argument("--engine", default="/home/user/opion/engine_log.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--matches", type=int, default=8)
    p.add_argument("--hands", type=int, default=400)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--wave", type=int, default=1)
    p.add_argument("--sb", type=int, default=0)
    p.add_argument("--quota", type=int, default=1_250_000)
    p.add_argument("--out-dir", default="/home/user/opion/data")
    p.add_argument("--timeout-s", type=int, default=2700)
    p.add_argument("--gzip", action="store_true")
    a = p.parse_args()

    raw = a.opponents
    if os.path.exists(raw):
        opponents = json.load(open(raw))
    else:
        opponents = json.loads(raw)
    tot_w = sum(o["w"] for o in opponents)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def opp_label(raw):
        if raw == "{SELF}":
            return "self"
        b = os.path.basename(raw)
        stem = os.path.splitext(b)[0]
        return os.path.basename(os.path.dirname(raw)) if stem in ("bot", "main") else stem

    def pick_opp(rnd):
        x = rnd.random() * tot_w
        for o in opponents:
            x -= o["w"]
            if x <= 0:
                return o["path"], opp_label(o["path"])
        return opponents[-1]["path"], opp_label(opponents[-1]["path"])

    t0 = time.monotonic()
    teacher_total = 0
    round_idx = 0
    per_round = []
    rng = random.Random(f"w{a.wave}:{a.sb}")

    def count_files(files):
        t_n = all_n = 0
        for s in files:
            if not os.path.exists(s):
                continue
            with open(s, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if '"t":"act"' in line or '"t":"disc"' in line:
                        all_n += 1
                        if '"is_t":true' in line:
                            t_n += 1
        return t_n, all_n

    while True:
        obs_files, procs, err_files = [], [], []
        for i in range(a.matches):
            seed = f"w{a.wave}:{a.sb}:r{round_idx}:m{i}"
            opp_path, opp_name = pick_opp(rng)
            opp = opp_path.replace("{SELF}", a.bot)
            teacher_as_a = (i % 2 == 0)
            ba, bb = (a.bot, opp) if teacher_as_a else (opp, a.bot)
            na, nb = (a.teacher_name, opp_name) if teacher_as_a else (opp_name, a.teacher_name)
            ld = os.path.join(a.out_dir, "logs", f"r{round_idx}_m{i}")
            os.makedirs(ld, exist_ok=True)
            obs = os.path.join(a.out_dir, f"obs_r{round_idx}_m{i}.jsonl")
            meta = os.path.join(a.out_dir, f"meta_r{round_idx}_m{i}.jsonl")
            obs_files.append(obs)
            soft = os.path.join(ld, "soft.jsonl")
            cmd = [a.python, a.engine, ba, bb,
                   "--stage", str(a.stage), "--hands", str(a.hands),
                   "--seed", seed, "--log-dir", ld,
                   "--obs-log", obs, "--meta-log", meta,
                   "--name-a", na, "--name-b", nb,
                   "--teacher-name", a.teacher_name]
            ef = open(os.path.join(ld, "engine_stderr.txt"), "wb")
            err_files.append(ef)
            env_i = dict(env)
            env_i["OPION_SOFT_LOG"] = soft
            procs.append(subprocess.Popen(cmd, env=env_i,
                                          stdout=subprocess.DEVNULL, stderr=ef))

        round_t0 = time.monotonic()
        deadline = round_t0 + min(a.timeout_s, 1500)
        hung = False
        for pr in procs:
            try:
                pr.wait(timeout=max(1.0, deadline - time.monotonic()))
            except Exception:
                hung = True
                for q in procs:
                    try:
                        q.terminate()
                    except Exception:
                        pass
        for ef in err_files:
            try:
                ef.close()
            except Exception:
                pass
        t_n, all_n = count_files(obs_files)
        teacher_total += t_n
        per_round.append({"round": round_idx, "teacher": t_n, "all": all_n,
                          "wall_s": round(time.monotonic() - round_t0, 1),
                          "hung": hung})
        print(f"[shard] round {round_idx}: teacher+{t_n} all+{all_n} "
              f"cum_teacher={teacher_total} wall={per_round[-1]['wall_s']}s "
              f"hung={hung}", flush=True)
        round_idx += 1
        if teacher_total >= a.quota:
            break
        if time.monotonic() - t0 > a.timeout_s:
            print("[shard] global watchdog", flush=True)
            break

    wall = time.monotonic() - t0
    ms_vals = []
    mode_n = {"argmax": 0, "sample": 0, "rule": 0}
    for f in glob.glob(os.path.join(a.out_dir, "logs", "*", "soft.jsonl")):
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    mode_n[rec.get("mode", "?")] = mode_n.get(rec.get("mode", "?"), 0) + 1
                except Exception:
                    pass
    summary = {
        "teacher_decisions": teacher_total, "rounds": round_idx,
        "all_records": sum(r["all"] for r in per_round),
        "wall_s": round(wall, 1), "dec_per_s": round(teacher_total / wall, 1),
        "per_round": per_round, "soft_modes": mode_n,
        "quota": a.quota, "matches": a.matches, "hands": a.hands,
        "wave": a.wave, "sb": a.sb, "host_cpus": os.cpu_count(),
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if a.gzip:
        for s in glob.glob(os.path.join(a.out_dir, "obs_r*_m*.jsonl")) + \
                 glob.glob(os.path.join(a.out_dir, "meta_r*_m*.jsonl")):
            with open(s, "rb") as fin, gzip.open(s + ".gz", "wb") as fout:
                fout.writelines(fin)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
