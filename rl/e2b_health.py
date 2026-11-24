#!/usr/bin/env python3
"""E2B health check: exit 0 iff the test sandbox connects AND runs a
trivial command. Re-pauses the sandbox either way."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


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
    try:
        sb = Sandbox.connect(ids["test"], timeout=90)
        r = sb.commands.run("echo ok", timeout=30)
        ok = (r.exit_code == 0 and "ok" in r.stdout)
    except Exception as e:
        print(f"E2B_DOWN {e!r}"[:160], flush=True)
        sys.exit(1)
    try:
        sb.beta_pause()
    except Exception:
        pass
    if ok:
        print("E2B_HEALTHY", flush=True)
        sys.exit(0)
    print("E2B_DEGRADED", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
