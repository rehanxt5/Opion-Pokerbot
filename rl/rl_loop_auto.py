#!/usr/bin/env python3
"""Continuous RL loop (2026-11-15, unattended): rollout round on the 8
sandboxes (rl_iterN.py: rollouts + screen of the current policy + fuel tar
for the next PPO run) -> T4 PPO leg via the colab CLI (ppo_auto cells) ->
verify + install ckpts -> next round. Time-boxed for the 23:59 IST
deadline: a full cycle only launches if projected round+T4+verify fits
before ABS_CUTOFF 23:10 IST; a screen-only round launches if only it fits.
Round completion = process exit + fuel tar on disk (the scripts' printed
DONE sentinel is stale from the sed chain). Trip-wire: two consecutive
hands-weighted strong-opp screen aggregates <= -0.0045 freeze the loop.
Never touches the frozen gen-8.6 submission; sandboxes are paused (never
killed) by the round scripts themselves."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VENV = os.path.join(ROOT, ".venv/bin/python")
AUTO = os.path.join(HERE, "ppo_auto")
LOOPLOG = "/tmp/rl_loop.log"
DEADLINE_HM = (23, 10)          # ABS_CUTOFF
ROUND_MIN = 42                  # projected wall of one rollout round
T4_MIN = 10                     # projected wall of one T4 leg incl verify
TRIPWIRE = -0.0045
PATH2OPP = {
    "bot/elite_instr.py": "elitev6",
    "opponents/gen86_pkg/package_g18_6/bot.py": "gen86",
    "opponents/elite83.py": "elite83",
    "opponents/styles/maniac.py": "maniac",
    "opponents/styles/station.py": "station",
    "opponents/styles/rock.py": "rock",
    "opponents/styles/randombot.py": "random",
    "opponents/styles/mixer.py": "mixer",
    "SELF": "SELF",
}
# (script, exec-timeout, completion sentinel)
CELLS = [
    ("c1_runtime.py", 540, "CELL_A_DONE"),
    ("c2_extract.py", 120, "CELL_B_DONE"),
    ("c3_model.py", 180, "CELL_C_DONE"),
    ("c4_gate.py", 300, "CELL_D_DONE"),
    ("c5_gae.py", 600, "CELL_E_DONE"),
    ("c6_probe.py", 180, "CELL_F_DONE"),
    ("c7_ppo.py", 1800, "CELL_G_DONE"),
    ("c8_save.py", 300, "PPO_DONE"),
]
AGGS = []  # strong-opp screen aggregates, in order


def log(msg):
    line = f"[LOOP {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOOPLOG, "a") as f:
        f.write(line + "\n")


def mins_to_deadline():
    now = time.localtime()
    dl = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                      DEADLINE_HM[0], DEADLINE_HM[1], 0, 0, 0, -1))
    return (dl - time.time()) / 60.0


def sh(cmd, timeout):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def round_errors(n):
    txt = open(f"/tmp/rl_iter{n}_run.log", errors="replace").read()
    m = re.findall(r"errors=\[([^\]]*)\]", txt)
    return m[-1].strip() if m else "?"


def wait_round(n, proc=None):
    """proc=None => externally launched (detect via tar + summary json)."""
    tar = f"/tmp/rl_iter_{n:04d}.tar"
    summ = os.path.join(ROOT, f"opion/data/rl/iter_{n:04d}_summary.json")
    log(f"waiting for rollout round {n}")
    t0 = time.time()
    while True:
        exited = (proc.poll() is not None) if proc is not None else (
            os.path.exists(tar) and os.path.exists(summ))
        if exited and os.path.exists(tar):
            errs = round_errors(n)
            log(f"round {n} done ({(time.time()-t0)/60:.0f} min) "
                f"errors=[{errs}] tar={os.path.getsize(tar)/1e6:.1f}MB")
            return True
        if exited:
            log(f"round {n} exited WITHOUT tar — failed")
            return False
        if mins_to_deadline() < 5:
            log(f"round {n} wait abandoned at deadline edge")
            return False
        if time.time() - t0 > 80 * 60:
            log(f"round {n} wait timed out (80 min)")
            return False
        time.sleep(40)


def parse_screen(n):
    """Screen of the policy that PLAYED round n (= iteration n-1)."""
    txt = open(f"/tmp/rl_iter{n}_run.log", errors="replace").read()
    seg = txt[txt.find("POLICY SCREEN"):]
    if "SCREEN_DONE" in seg:
        seg = seg[:seg.find("SCREEN_DONE")]
    per = {}
    for m in re.finditer(r"^\s+(\S+)\s+([+-]\d+\.\d+)\s+±\s+([+-]?\d+\.\d+)"
                         r"\s+\((\d+) matches, ([\d,]+) hands\)", seg, re.M):
        opp = PATH2OPP.get(m.group(1), m.group(1))
        per[opp] = (float(m.group(2)), int(m.group(5).replace(",", "")))
    agg = None
    tot_v = tot_n = 0.0
    for opp in ("elitev6", "gen86", "elite83"):
        if opp in per:
            m, h = per[opp]
            tot_v += m * h
            tot_n += h
    if tot_n:
        agg = tot_v / tot_n
    return per, agg


def gen_round(n):
    prev = n - 1
    src = os.path.join(HERE, f"rl_iter{prev}.py")
    dst = os.path.join(HERE, f"rl_iter{n}.py")
    subs = [(f"rl_iter{prev}", f"rl_iter{n}"), (f"rl:i{prev}", f"rl:i{n}"),
            (f"iter_{prev:04d}", f"iter_{n:04d}"), (f"iter{prev}", f"iter{n}"),
            (f"ITERATION-{prev}", f"ITERATION-{n}"),
            (f"ITER-{prev}", f"ITER-{n}"),
            (f"places iter-{prev}", f"places iter-{n}"),
            ("RL_ITER15_DONE", f"RL_ITER{n}_DONE")]
    s = open(src).read()
    for a, b in subs:
        s = s.replace(a, b)
    open(dst, "w").write(s)
    stale = sum(s.count(x) for x in (f"rl:i{prev}", f"iter_{prev:04d}",
                                     f"iter{prev}"))
    assert stale == 0, f"stale refs in {dst}: {stale}"
    log(f"generated rl_iter{n}.py (stale refs: 0)")


def launch_round(n):
    lf = f"/tmp/rl_iter{n}_run.log"
    f = open(lf, "w")
    p = subprocess.Popen([VENV, os.path.join(HERE, f"rl_iter{n}.py")],
                         stdout=f, stderr=subprocess.STDOUT)
    log(f"launched round {n} pid={p.pid} -> {lf}")
    return p


def t4_leg(n):
    """Train PPO iteration n from /tmp/rl_iter_{n:04d}.tar on a fresh T4."""
    tar = f"/tmp/rl_iter_{n:04d}.tar"
    outdir = f"/tmp/iter{n}_out"
    for attempt in (1, 2, 3):
        t0 = time.time()
        try:
            assert os.path.exists(tar), f"missing {tar}"
            try:
                sh("colab stop -s opion-t4", 60)
            except Exception:
                pass
            r = sh("colab new -s opion-t4 --gpu T4", 300)
            if "Session READY" not in r.stdout + r.stderr:
                raise RuntimeError(f"colab new: {(r.stdout + r.stderr)[-200:]}")
            st = sh("colab status -s opion-t4", 60)
            if "Hardware: T4" not in st.stdout + st.stderr:
                raise RuntimeError(f"not a T4: {(st.stdout + st.stderr)[-160:]}")
            up = sh(f"colab upload {tar} /content/rl_iter.tar -s opion-t4",
                    420)
            if "Uploaded" not in up.stdout + up.stderr:
                raise RuntimeError(f"upload failed: {(up.stdout+up.stderr)[-160:]}")
            hashes, result_line = {}, None
            for fn, tmo, sentinel in CELLS:
                r = sh(f"colab exec -s opion-t4 -f {AUTO}/{fn} "
                       f"--timeout {tmo}", tmo + 150)
                blob = r.stdout + r.stderr
                if sentinel not in blob:
                    raise RuntimeError(f"{fn}: no {sentinel}: …{blob[-400:]}")
                if fn == "c8_save.py":
                    mm = re.search(r"^RESULT (\{.*\})$", blob, re.M)
                    if not mm:
                        raise RuntimeError(f"{fn}: no RESULT line")
                    result_line = mm.group(1)
                    for hm in re.finditer(r"^(policy_\w+\.\w+)\s+(\d+) bytes"
                                          r"\s+sha256=([0-9a-f]{16})$", blob,
                                          re.M):
                        hashes[hm.group(1)] = (hm.group(3), int(hm.group(2)))
            shutil.rmtree(outdir, ignore_errors=True)
            os.makedirs(outdir)
            for remote, local in (("policy_np.npz", "policy_np.npz"),
                                  ("policy_rl.pt", "policy_rl.pt")):
                dl = sh(f"colab download /content/{remote} "
                        f"{outdir}/{local} -s opion-t4", 240)
                if "Downloaded" not in dl.stdout + dl.stderr:
                    raise RuntimeError(f"download {remote} failed")
            ok, note = verify_install(n, outdir, hashes)
            if not ok:
                raise RuntimeError(f"verify failed: {note}")
            log(f"T4 leg {n} OK in {(time.time()-t0)/60:.1f} min | "
                f"RESULT {result_line} | {note}")
            try:
                sh("colab stop -s opion-t4", 90)
            except Exception:
                pass
            return json.loads(result_line)
        except Exception as e:
            log(f"T4 leg {n} attempt {attempt} FAILED: {repr(e)[:280]}")
            try:
                sh("colab stop -s opion-t4", 60)
            except Exception:
                pass
            time.sleep(45)
    log(f"T4 leg {n} GAVE UP after 3 attempts — RL frozen at iter-{n-1}")
    return None


def verify_install(n, outdir, hashes):
    import zipfile
    import numpy as np
    npz = os.path.join(outdir, "policy_np.npz")
    pt = os.path.join(outdir, "policy_rl.pt")
    md = -1.0
    for f, key in ((npz, "policy_np.npz"), (pt, "policy_rl.pt")):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if key in hashes and not h.startswith(hashes[key][0]):
            return False, f"{key} sha256 mismatch"
        if key in hashes and os.path.getsize(f) != hashes[key][1]:
            return False, f"{key} size mismatch"
    z = np.load(npz)
    npar = sum(int(np.prod(v.shape)) for v in z.values())
    nans = sum(int(np.isnan(np.asarray(v, dtype=np.float32)).sum())
               for v in z.values())
    if len(z.files) != 60 or npar != 812713 or nans:
        return False, f"npz bad: {len(z.files)} keys {npar:,} params {nans} nans"
    prev = os.path.join(ROOT, f"opion/rl/ckpts/policy_np_iter{n-1}.npz")
    if os.path.exists(prev):
        zo = np.load(prev)
        if set(zo.files) != set(z.files):
            return False, "key set changed vs previous"
        md = max(float(np.max(np.abs(z[k].astype(np.float64)
                                     - zo[k].astype(np.float64))))
                 for k in z.files)
        if md > 0.05:
            return False, f"weight jump too large: {md:.4f}"
    zipfile.ZipFile(pt).testzip()
    shutil.copy(npz, os.path.join(ROOT, f"opion/rl/ckpts/policy_np_iter{n}.npz"))
    shutil.copy(pt, os.path.join(ROOT, f"opion/rl/ckpts/policy_rl_iter{n}.pt"))
    return True, (f"60 keys {npar:,} params max|dW|={md:.4f}" if md >= 0
                  else f"60 keys {npar:,} params")


def main():
    log(f"LOOP_START now={time.strftime('%H:%M %Z')} "
        f"deadline-margin={mins_to_deadline():.0f} min")
    if not wait_round(16):            # externally launched, already done
        log("LOOP_END round16 failed — no training this cycle")
        return
    per, agg = parse_screen(16)
    AGGS.append(agg)
    log(f"SCREEN iter-15 (round 16): {json.dumps(per)} strong-agg={agg:+.5f}")
    if t4_leg(16) is None:
        log("LOOP_END T4-16 failed")
        return
    n = 17
    while True:
        margin = mins_to_deadline()
        if len(AGGS) >= 2 and all(a is not None and a <= TRIPWIRE
                                  for a in AGGS[-2:]):
            log(f"FREEZE_TRIPWIRE: aggregates {AGGS[-2:]}")
            break
        if margin < ROUND_MIN + T4_MIN + 5:
            if margin >= ROUND_MIN + 5:
                gen_round(n)
                p = launch_round(n)
                if wait_round(n, p):
                    per, agg = parse_screen(n)
                    AGGS.append(agg)
                    log(f"SCREEN-ONLY iter-{n-1} (round {n}): "
                        f"{json.dumps(per)} strong-agg={agg:+.5f}")
            log(f"FREEZE_TIME: {mins_to_deadline():.0f} min left — loop over")
            break
        gen_round(n)
        p = launch_round(n)
        if not wait_round(n, p):
            log(f"LOOP_END round {n} failed — freezing at iter-{n-1}")
            break
        per, agg = parse_screen(n)
        AGGS.append(agg)
        log(f"SCREEN iter-{n-1} (round {n}): {json.dumps(per)} "
            f"strong-agg={agg:+.5f}")
        if mins_to_deadline() < T4_MIN + 5:
            log(f"FREEZE_TIME: no T4 leg for tar-{n} "
                f"({mins_to_deadline():.0f} min left)")
            break
        if t4_leg(n) is None:
            log(f"LOOP_END T4-{n} failed — freezing at iter-{n-1}")
            break
        n += 1
    log(f"LOOP_END aggregates={AGGS}")


if __name__ == "__main__":
    main()
