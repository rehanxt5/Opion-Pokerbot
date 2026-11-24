#!/usr/bin/env python3
"""Map pulled shard archives to their true sandbox index (sbNN) using the
seed embedded in the first obs record ("w1:{sb}:r0:m0"), then rename the
candidate dirs. Safe to re-run as more shards land."""
import glob
import gzip
import io
import json
import os
import re
import sys
import tarfile

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "opion", "data", "fleet25m", "tookover_w1")


def seed_sb(tgz_path):
    with tarfile.open(tgz_path, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if re.search(r"obs_r0_m0\.jsonl(\.gz)?$", m.name)]
        if not members:
            return None
        m = sorted(members, key=lambda x: x.size)[0]
        f = tf.extractfile(m)
        raw = f.read(1 << 20)
        if m.name.endswith(".gz"):
            raw = gzip.decompress(raw + b"") if raw[:2] == b"\x1f\x8b" else raw
        line = raw.split(b"\n", 1)[0].decode(errors="replace")
        hit = re.search(r'"seed":"w1:(\d+):', line)
        return int(hit.group(1)) if hit else None


def main():
    mapping = {}
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        if re.match(r"^sb\d+$", base):
            continue
        tgz = os.path.join(d, "shard.tgz")
        if not os.path.exists(tgz):
            continue
        try:
            sb = seed_sb(tgz)
        except Exception as e:
            print(f"{base}: ERROR {repr(e)[:120]}")
            continue
        if sb is None:
            print(f"{base}: no seed found — skipped")
            continue
        new = os.path.join(OUT, f"sb{sb:02d}")
        if os.path.exists(new):
            print(f"{base}: sb{sb:02d} target exists — skipped")
            continue
        os.rename(d, new)
        mapping[base] = f"sb{sb:02d}"
        print(f"{base} -> sb{sb:02d}")
    total = 0
    for s in glob.glob(os.path.join(OUT, "sb*", "summary.json")):
        try:
            total += json.load(open(s)).get("teacher_decisions", 0)
        except Exception:
            pass
    n = len(glob.glob(os.path.join(OUT, "sb*", "shard.tgz")))
    print(f"mapped_now={json.dumps(mapping)} shards_renamed={n} "
          f"local_teacher_total={total}")


if __name__ == "__main__":
    main()
