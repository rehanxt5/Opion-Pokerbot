#!/usr/bin/env python3
"""Pre-RL checklist item 4a: SHA256 manifest of everything irreplaceable.

Hashes the 40 local packed shards (acts.npz / disc.npz / manifest.json each)
plus the two trained model files. Writes:
  opion/data/packed/SHA256SUMS.json   (structured: size, sha256, mtime)
  opion/data/packed/SHA256SUMS.txt    (sha256sum -c compatible)
Spot re-verifies 3 random files before declaring success.
"""
import hashlib
import json
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKED = os.path.join(ROOT, "opion", "data", "packed")
TRAIN = os.path.join(ROOT, "opion", "train")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    targets = []
    for w in ("w1", "w2"):
        for sb in range(20):
            for fn in ("acts.npz", "disc.npz", "manifest.json"):
                p = os.path.join(PACKED, w, f"sb{sb:02d}", fn)
                if os.path.exists(p):
                    targets.append(p)
    for fn in ("final_act.pt", "final_disc.pt"):
        p = os.path.join(TRAIN, fn)
        if os.path.exists(p):
            targets.append(p)
    assert len(targets) >= 122, f"expected >=122 files, found {len(targets)}"

    t0 = time.monotonic()
    entries = []
    for p in targets:
        st = os.stat(p)
        entries.append({"path": os.path.relpath(p, ROOT),
                        "bytes": st.st_size,
                        "sha256": sha256(p)})
    total = sum(e["bytes"] for e in entries)
    print(f"[sums] hashed {len(entries)} files, {total / 1e6:.1f}MB "
          f"in {time.monotonic() - t0:.1f}s", flush=True)

    with open(os.path.join(PACKED, "SHA256SUMS.json"), "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "files": len(entries), "total_bytes": total,
                   "entries": entries}, f, indent=1)
    with open(os.path.join(PACKED, "SHA256SUMS.txt"), "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {e['path']}\n")

    # spot re-verify 3 random files
    rng = random.Random(7)
    for e in rng.sample(entries, 3):
        ok = sha256(os.path.join(ROOT, e["path"])) == e["sha256"]
        print(f"[sums] re-verify {e['path']}: {'OK' if ok else 'FAIL'}",
              flush=True)
        assert ok
    print(f"CHECKSUMS_DONE {len(entries)} files {total / 1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
