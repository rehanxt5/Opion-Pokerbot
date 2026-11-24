# PPO auto cell B: extract rl_iter.tar (notebook cell 4)
import glob
import os
import tarfile

assert os.path.exists("/content/rl_iter.tar"), "upload rl_iter.tar first"
with tarfile.open("/content/rl_iter.tar") as tf:
    tf.extractall("/content")
names = sorted(os.listdir("/content/rl_iter"))
print("rl_iter/:", [n for n in names if not n.startswith("rollouts")],
      f"+ {len(glob.glob('/content/rl_iter/rollouts/*.npz'))} rollout files")
print("beta =", open("/content/rl_iter/beta.txt").read().strip())
import numpy as np
z = np.load("/content/rl_iter/rollouts/sb0_m0000.npz")
print("shard keys:", sorted(z.files), "| rows:", len(z["bucket"]))
print("CELL_B_DONE", flush=True)
