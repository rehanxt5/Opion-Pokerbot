import json
import os
import subprocess
import sys
import threading
import time

os.chdir("/content")

print("python:", sys.version.split()[0])
import numpy as np
print("numpy:", np.__version__)
print("vm cores:", os.cpu_count())

t0 = time.monotonic()
r = subprocess.run(["python3", "bot.py"], stdin=subprocess.DEVNULL,
                   capture_output=True, text=True, timeout=60)
print(f"startup+EOF: {time.monotonic() - t0:.3f}s rc={r.returncode} "
      f"stderr={len(r.stderr)}B")

ACT = {"type": "action_request", "hand_id": 1, "street": "flop",
       "board": ["Ah", "7d", "2s"], "pot": 24, "my_bet": 0, "opp_bet": 6,
       "to_call": 6, "my_stack": 188, "opp_stack": 182,
       "legal_actions": ["fold", "call", "raise"], "min_raise": 12,
       "max_raise": 188, "raises_left": 3,
       "history": [{"street": "preflop", "actor": "opp", "action": "raise",
                    "amount": 6}],
       "time_left_ms": 29000}
DISC = {"type": "discard_request", "hand_id": 1,
        "hole_cards": ["As", "Kd", "7c"], "board": ["Ah", "7d", "2s"],
        "pot": 24, "my_stack": 188, "opp_stack": 182,
        "time_left_ms": 29000}

p = subprocess.Popen(["taskset", "-c", "0", "python3", "bot.py"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, bufsize=1)
time.sleep(1.5)
lat = []
for i in range(300):
    m = ACT if i % 5 < 4 else DISC
    t0 = time.monotonic()
    p.stdin.write(json.dumps(m) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    lat.append((time.monotonic() - t0) * 1000.0)
    assert line.strip(), "empty reply"
p.stdin.close()
p.wait(timeout=10)
lat.sort()
print(f"latency probe on 1 pinned core: n={len(lat)} "
      f"mean={sum(lat) / len(lat):.2f}ms median={lat[len(lat) // 2]:.2f}ms "
      f"p99={lat[int(len(lat) * 0.99)]:.2f}ms max={lat[-1]:.2f}ms "
      f"(limits: 5000ms/reply, 30000ms bank per leg)")
print(f"probe bot stderr: {len(p.stderr.read())}B")

peak = [0]
stop = [False]


def samp():
    while not stop[0]:
        try:
            out = subprocess.run(["ps", "-eo", "rss=,comm="],
                                 capture_output=True, text=True).stdout
            tot = sum(int(l.split()[0]) for l in out.splitlines()
                      if "python" in l)
            peak[0] = max(peak[0], tot)
        except Exception:
            pass
        time.sleep(0.3)


th = threading.Thread(target=samp, daemon=True)
th.start()

wins = 0
for s in (1, 2, 3, 4, 5):
    cmd = ["taskset", "-c", "0", "python3", "engine.py", "bot.py",
           "random_bot.py", "--seed", str(s), "--log-dir", f"logs_{s}"]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        w = time.monotonic() - t0
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        print(f"--- match seed {s}: wall {w:.1f}s rc={r.returncode}")
        for l in lines[-5:]:
            print("   ", l)
        if "Result: A wins" in r.stdout:
            wins += 1
        if r.stderr.strip():
            print("    ENGINE STDERR:", r.stderr[-200:])
        sb = sum(os.path.getsize(f) for f in
                 (f"logs_{s}/A_leg1_stderr.txt", f"logs_{s}/A_leg2_stderr.txt")
                 if os.path.exists(f))
        nz = sum(1 for l in lines if "notices" in l and "notices 0" not in l)
        print(f"    bot stderr bytes: {sb}")
    except Exception as e:
        print(f"--- match seed {s} FAILED: {e!r}")

stop[0] = True
time.sleep(0.6)
print(f"MATCHES: bot won {wins}/5 vs random")
print(f"PEAK python RSS (engine + 2 bot procs): {peak[0] / 1024:.0f} MB "
      f"of the 1024 MB budget")
print("REMOTE_TEST_DONE")
