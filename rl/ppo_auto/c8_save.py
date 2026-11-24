# PPO auto cell H: save + export + integrity (notebook cell 10, headless:
# no google.colab files.download -- artifacts pulled via `colab download`)
import hashlib
import json as _json
import os
import time

KL_STAR = 0.10
new_beta = min(10.0, max(0.01, BETA * float(np.exp(2 * (kl_probe - KL_STAR) / KL_STAR))))
RESULT = {"kl_probe": round(kl_probe, 4), "beta_next": round(new_beta, 4),
          "decisions": int(N), "epochs": EPOCHS, "lr": LR,
          "wall_min": round((time.time() - t0) / 60, 1),
          "probe_mix": [round(float(x), 4) for x in pmix]}
print("RESULT " + _json.dumps(RESULT))
torch.save({"model": model.state_dict(), "result": RESULT}, "/content/policy_rl.pt")
np_policy.save_npz({k: v.detach().cpu().numpy()
                    for k, v in model.state_dict().items()},
                   "/content/policy_np.npz")
for f in ("/content/policy_np.npz", "/content/policy_rl.pt"):
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    print(f"{os.path.basename(f)}  {os.path.getsize(f)} bytes  sha256={h[:16]}")
print("PPO_DONE", flush=True)
