"""Run serial matched C1/D4 graph comparisons on the original checkpoint."""

import argparse
import json
import subprocess
from pathlib import Path

from continue_targets import ROOT, PY, environment
from common import write_json
from mini_probe import idle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--roles",
        nargs="+",
        choices=("control", "ngram", "mtp"),
        default=["control", "ngram", "mtp"],
    )
    p.add_argument("--reference", type=Path)
    p.add_argument("--native-metrics", action="store_true", help="Run two 1K/8K streaming rounds with 1,024 output tokens")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs = {
        c["name"]: c
        for c in json.loads((ROOT / "configs/experiments.json").read_text())[
            "experiments"
        ]
    }
    names = {
        "control": "deterministic-control-triton",
        "ngram": "deterministic-ngram-triton",
        "mtp": "deterministic-eagle-triton",
    }
    ref = args.reference or args.output / "control/result.json"
    phases = {}
    for role in args.roles:
        idle(30000)
        trial = args.output / role
        trial.mkdir()
        cfg = configs[names[role]]
        env = environment("target")
        env.update(cfg["env"])
        if role != "control":
            env["SGLANG_OPT_MHA_VERIFY_GRAPH"] = "1"
        cmd = [PY, "bench/graph_spec_screen.py", "--output", str(trial / "result.json")]
        if args.native_metrics:
            cmd = [PY, "bench/target_metrics.py", "--output", str(trial / "metrics.json"), "--repeats", "2"]
        elif role != "control":
            cmd += ["--reference", str(ref)]
        plan = trial / "plan.json"
        write_json(plan, [{"command": cmd, "timeout_s": 900}])
        flags = [*cfg["flags"], "--enable-metrics", "--max-total-tokens", "16384"]
        with (trial / "launcher.log").open("x") as log:
            run = subprocess.run(
                [
                    PY,
                    "bench/mini_probe.py",
                    "--profile",
                    cfg["profile"],
                    "--output",
                    str(trial / "probe"),
                    "--validation-plan",
                    str(plan),
                    "--",
                    *flags,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        phases[role] = run.returncode
        write_json(args.output / "runs.json", phases)
        if role == "control" and run.returncode:
            break
    return int(any(phases.values()))


if __name__ == "__main__":
    raise SystemExit(main())
