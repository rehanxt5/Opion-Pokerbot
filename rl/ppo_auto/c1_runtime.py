# PPO auto cell A: runtime check + packed tether data (notebook cells 1+3)
import subprocess
import sys

print(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                     text=True).stdout.strip(), flush=True)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0), flush=True)
assert torch.cuda.is_available(), "no CUDA on this session"

import glob
import os
import tarfile
if not glob.glob("/content/packed/*/*/acts.npz"):
    if not os.path.exists("/content/packed.tar"):
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "gdown"], check=True)
            import gdown
        gdown.download(id="1AgcO3IHlhFc4czzr1fU_EG39Bh7809TA",
                       output="/content/packed.tar")
    with tarfile.open("/content/packed.tar") as tf:
        tf.extractall("/content")
print(len(glob.glob("/content/packed/*/*/acts.npz")), "shards", flush=True)
print("CELL_A_DONE", flush=True)
