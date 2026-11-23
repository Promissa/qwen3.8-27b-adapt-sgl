#!/usr/bin/env python3
"""Run independent lossless A-layer trials serially, retaining failed gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import ROOT, now, write_json

DEFAULTS = ["gemm-cutlass", "gemm-cudnn", "gdn-decode-flashinfer",
            "gdn-prefill-flashinfer", "graph-precarve", "host-checkpoints"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", choices=["present", "absent"], required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reference-tokens", type=Path, required=True)
    parser.add_argument("names", nargs="*", default=DEFAULTS)
    args = parser.parse_args()
    configs = {c["name"]: c for c in json.loads((ROOT / "configs/experiments.json").read_text())["experiments"]}
    for name in args.names:
        if name not in configs or configs[name]["gate"] != "token-equality":
            parser.error(f"{name}: this batch requires a lossless token-equality gate")
        if configs[name].get("applicability") == "source_inactive":
            parser.error(f"{name}: {configs[name]['not_applicable_reason']}")
        if (ROOT / "results/experiments" / f"{args.prefix}-{name}").exists():
            parser.error(f"Existing evidence for {name}; use a new prefix")
    output = ROOT / "results" / f"{args.prefix}-batch.json"
    if output.exists():
        parser.error("Batch manifest already exists")
    result = {"at": now(), "status": "running", "planned": args.names, "trials": []}
    write_json(output, result)
    try:
        for name in args.names:
            tag = f"{args.prefix}-{name}"
            result["current"] = tag
            write_json(output, result)
            cmd = [sys.executable, str(ROOT / "bench/run_experiment.py"), name,
                   "--tag", tag, "--desktop", args.desktop,
                   "--reference-tokens", str(args.reference_tokens.resolve())]
            with (ROOT / "results" / f"{tag}.log").open("w") as log:
                completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            probe_path = ROOT / "results/experiments" / tag / "probe/probe.json"
            if not probe_path.exists():
                raise RuntimeError(f"Trial driver failed without probe evidence: {tag}")
            probe = json.loads(probe_path.read_text())
            result["trials"].append({"name": name, "tag": tag, "returncode": completed.returncode,
                "status": probe["status"], "capacity_gate_passed": probe.get("capacity_gate_passed"),
                "error": probe.get("error"), "evidence": str(probe_path)})
            write_json(output, result)
            if probe["status"] == "running" or "cannot be bound" in (probe.get("error") or ""):
                raise RuntimeError(f"Infrastructure failure needs inspection before another launch: {tag}")
        result["status"] = "completed"
        result.pop("current", None)
    except Exception as e:
        result.update(status="stopped", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        write_json(output, result)


if __name__ == "__main__":
    main()
