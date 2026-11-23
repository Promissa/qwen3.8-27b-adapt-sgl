#!/usr/bin/env python3
"""Run one ablation in an owned server, recording the protocol before execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, now, write_json
from eval_quality import DEFAULT_MAX_TOKENS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--desktop", choices=["present", "absent"], required=True)
    parser.add_argument("--quality", action="store_true", help="Also run full GSM8K and the quality battery")
    parser.add_argument("--hard-quality", choices=["battery", "gpqa"], default="battery")
    parser.add_argument("--quality-max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Thinking output cap; must match the quality control")
    parser.add_argument("--reference-tokens", type=Path,
                        help="Require greedy token equality against this control capture")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.quality_max_tokens <= 0:
        parser.error("--quality-max-tokens must be positive")
    if extra[:1] == ["--"]:
        extra = extra[1:]
    configs = json.loads((ROOT / "configs/experiments.json").read_text())["experiments"]
    matches = [x for x in configs if x["name"] == args.name]
    if not matches:
        parser.error("Unknown experiment; see configs/experiments.json")
    experiment = matches[0]
    if experiment.get("applicability") == "source_inactive" and not args.dry_run:
        parser.error(experiment["not_applicable_reason"] + " See " + experiment["source_audit"])
    output = ROOT / "results/experiments" / args.tag
    py = sys.executable
    steps = [{"command": [py, str(ROOT / "bench/token_compare.py"), "--output", str(output / "tokens.json")]},
             {"command": [py, str(ROOT / "bench/api_smoke.py"), "--output", str(output / "api.json")]},
             {"command": ["bash", str(ROOT / "bench/run_serving_bench.sh"), args.tag]}]
    if args.reference_tokens:
        steps[0]["command"] += ["--reference", str(args.reference_tokens.resolve())]
    steps.append({"command": [py, str(ROOT / "bench/summarize_serving.py"), args.tag]})
    steps.append({"command": [py, str(ROOT / "bench/long_context.py"), "--near-limit",
                               "--output", str(output / "retrieval-near-limit.json")]})
    if args.quality:
        for benchmark in ("gsm8k",):
            steps.append({"command": [py, str(ROOT / "bench/eval_quality.py"), "--tag", args.tag,
                                       "--eval", benchmark, "--max-tokens", str(args.quality_max_tokens)]})
        if args.hard_quality == "gpqa":
            steps.append({"command": [py, str(ROOT / "bench/eval_quality.py"), "--tag", args.tag,
                                       "--eval", "gpqa", "--max-tokens", str(args.quality_max_tokens)]})
        else:
            steps.append({"command": [py, str(ROOT / "bench/quality_battery.py"),
                                       "--output", str(output / "quality-battery.json")]})
    env = dict(os.environ)
    for cfg in configs:
        for key in cfg["env"]:
            env.pop(key, None)
    env.update(experiment["env"])
    env.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT))
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    env.setdefault("MAX_JOBS", "2")
    cmd = [py, str(ROOT / "bench/probe.py"), "--profile", experiment["profile"],
           "--desktop", args.desktop, "--output", str(output / "probe"),
           "--validation-plan", str(output / "validation-plan.json"), "--"] + experiment["flags"] + extra
    manifest = {"at": now(), "experiment": experiment, "command": cmd, "validation": steps,
                "quality_requested": args.quality, "promotion_status": "unvalidated"}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment.json", manifest)
    write_json(output / "validation-plan.json", steps)
    # Failed experiments retain manifests and raw evidence and stop immediately.
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
