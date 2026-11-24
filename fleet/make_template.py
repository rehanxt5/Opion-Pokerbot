#!/usr/bin/env python3
"""Build the opion dataset-generation template on E2B (Hobby tier).

Programmatic template build (daemonless, no Docker): python3.11 base + numpy,
teacher bot + logging engine + shard runner baked in, 8 vCPU / 8GB resources.
Writes the template id to fleet/template.txt.
"""
import os
import sys
import time

FLEET = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(FLEET))


def load_key():
    for line in open(os.path.join(REPO, ".env")):
        line = line.strip()
        if line.startswith("E2B_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("E2B_KEY not found in .env")


def main():
    os.environ["E2B_API_KEY"] = load_key()
    os.chdir(FLEET)  # copy() sources are relative to the build context
    from e2b import Template, default_build_logger

    # stage the opponents tree into one tarball for a single-copy upload
    os.system("tar -czf opp.tgz opponents")
    print("opponents tar:", os.path.getsize(os.path.join(FLEET, "opp.tgz")), "bytes")

    template = (
        Template()
        .from_python_image("3.11")
        .pip_install(["numpy"])
        .copy("teacher/elite_instr.py", "/home/user/opion/bot/elite_instr.py")
        .copy("engine_log.py", "/home/user/opion/engine_log.py")
        .copy("run_shard.py", "/home/user/opion/run_shard.py")
        .copy("qc_report.py", "/home/user/opion/qc_report.py")
        .copy("pairs.json", "/home/user/opion/pairs.json")
        .copy("opp.tgz", "/home/user/opion/opp.tgz")
        .run_cmd("mkdir -p /home/user/opion && tar -xzf /home/user/opion/opp.tgz -C /home/user/opion "
                 "&& rm /home/user/opion/opp.tgz && python3 -m py_compile /home/user/opion/bot/elite_instr.py "
                 "/home/user/opion/engine_log.py /home/user/opion/run_shard.py "
                 "/home/user/opion/opponents/elite83.py /home/user/opion/opponents/styles/station.py "
                 "&& echo TmplCompile_OK", user="root")
    )

    t0 = time.monotonic()
    info = Template.build(
        template,
        "opion-gen-v1",
        cpu_count=8,
        memory_mb=8192,
        on_build_logs=default_build_logger(),
    )
    print(f"build done in {time.monotonic() - t0:.1f}s:", info)
    tid = getattr(info, "template_id", None) or str(info)
    with open(os.path.join(FLEET, "template.txt"), "w") as f:
        f.write("opion-gen-v1\n")
    print("TEMPLATE_ID", tid)


if __name__ == "__main__":
    main()
