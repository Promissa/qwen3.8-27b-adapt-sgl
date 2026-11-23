#!/usr/bin/env python3
"""Run the frozen application suite, throughput, capacity and text soak serially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from continue_targets import ROOT, PY, environment, idle, now, write_json


def scores(trial, benchmark, split):
    path = trial / (f"longbench-{split}.json" if benchmark == "longbench"
                    else f"bfcl-{split}/summary.json")
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if benchmark == "bfcl" and result["status"] == "failed":
        recovered = trial / f"bfcl-{split}-rescore/summary.json"
        if recovered.exists():
            result = json.loads(recovered.read_text())
    if result["status"] != "completed":
        return None
    if benchmark == "bfcl":
        return result["items"]
    return {case["id"]: case["correct"] for case in result["results"]}


def applications_report(output):
    summary = {}
    for benchmark in ("bfcl", "longbench"):
        reference = output / "reference"
        initial = scores(reference, benchmark, "screen")
        repeat = scores(reference, benchmark, "screen-repeat")
        summary[benchmark] = {}
        if initial is not None and repeat is not None and initial.keys() == repeat.keys():
            summary[benchmark]["reference_repeat_disagreements"] = [
                key for key in initial if initial[key] != repeat[key]]
        for profile in ("balanced", "max-context"):
            rows = {}
            for split in ("screen", "holdout"):
                control = scores(reference, benchmark, split)
                candidate = scores(output / profile, benchmark, split)
                if control is None or candidate is None or control.keys() != candidate.keys():
                    rows[split] = {"status": "incomplete"}
                    continue
                common = [key for key in control if control[key] is not None and candidate[key] is not None]
                rows[split] = {
                    "attempted": len(control), "total": len(common),
                    "reference_passed": sum(control[key] for key in common),
                    "candidate_passed": sum(candidate[key] for key in common),
                    "capacity_rejected_reference": [key for key in control if control[key] is None],
                    "capacity_rejected_candidate": [key for key in candidate if candidate[key] is None],
                    "regressions": [key for key in common if control[key] and not candidate[key]],
                    "improvements": [key for key in common if not control[key] and candidate[key]],
                }
            summary[benchmark][profile] = rows
    write_json(output / "paired-quality.json", summary)
    return summary


def recheck_manifests(manifest_dir, output):
    comparisons = applications_report(manifest_dir / "application")
    for benchmark, report in comparisons.items():
        targets = {}
        for profile in ("balanced", "max-context"):
            targets[profile] = {case for row in report[profile].values()
                                for case in row.get("regressions", [])}
        targets["reference"] = targets["balanced"] | targets["max-context"]
        original = json.loads((manifest_dir / f"{benchmark}-ids.json").read_text())
        for profile, cases in targets.items():
            if not cases:
                continue
            if benchmark == "bfcl":
                subset = dict(original, ids={category: [case for case in values if case in cases]
                                            for category, values in original["ids"].items()})
            else:
                subset = dict(original, items=[dict(row, screen=True) for row in original["items"]
                                               if row["id"] in cases])
            write_json(output / profile / f"{benchmark}-recheck-ids.json", subset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("application", "recheck"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    args.output.mkdir(parents=True)
    manifest_dir = args.manifests.resolve()
    output = args.output.resolve()
    state = {"started_at": now(), "pid": os.getpid(), "status": "running", "profiles": {}}
    if args.phase == "application":
        paths = [manifest_dir / name for name in ("longbench-ids.json", "bfcl-ids.json")]
        state["manifest_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in paths}
    else:
        recheck_manifests(manifest_dir, output)
    state_path = output / "runs.json"
    write_json(state_path, state)
    configs = [
        ("reference", "target", "balanced", 49152, 4, []),
        ("balanced", "target-embedding-nvfp4", "balanced", 49152, 4, []),
        ("max-context", "target-embedding-nvfp4", "max-context", 81920, 1, ["--mem-fraction-static", "0.92"]),
    ]
    cache = ROOT / ".cache/app-eval"
    for name, model, profile, pool, slots, flags in configs:
        trial = output / name
        if args.phase == "recheck" and name == "reference" and not trial.exists():
            continue
        idle(30000)
        trial.mkdir(exist_ok=True)
        weights = ROOT / "weights" / model
        pins = ["--enable-metrics", "--max-total-tokens", str(pool), "--max-mamba-cache-size", str(slots)]
        steps = []

        def step(*command, timeout, keep_going=False):
            steps.append({"command": list(command), "timeout_s": timeout,
                          "continue_on_failure": keep_going or args.phase == "application"})

        if args.phase == "recheck":
            bfcl_recheck = trial / "bfcl-recheck-ids.json"
            longbench_recheck = trial / "longbench-recheck-ids.json"
            if bfcl_recheck.exists():
                step(str(cache / "bfcl-venv/bin/python"), "bench/bfcl_subset.py", "--split", "all",
                     "--manifest", str(bfcl_recheck), "--output", str(trial / "bfcl-recheck"),
                     timeout=1800, keep_going=True)
            if longbench_recheck.exists():
                step(PY, "bench/longbench_subset.py", "--dataset", str(cache / "longbench-v2.json"),
                     "--template", str(cache / "longbench-0shot.txt"), "--model-path", str(weights),
                     "--manifest", str(longbench_recheck), "--output", str(trial / "longbench-recheck.json"),
                     timeout=900, keep_going=True)
        else:
            step(PY, "bench/api_smoke.py", "--output", str(trial / "api.json"), timeout=180)
            subsets = [("screen", "screen"), ("holdout", "holdout")]
            if name == "reference":
                subsets.insert(1, ("screen", "screen-repeat"))
            for split, tag in subsets:
                step(PY, "bench/longbench_subset.py", "--dataset", str(cache / "longbench-v2.json"),
                     "--template", str(cache / "longbench-0shot.txt"), "--model-path", str(weights),
                     "--manifest", str(manifest_dir / "longbench-ids.json"), "--split", split,
                     "--output", str(trial / f"longbench-{tag}.json"), timeout=900)
                step(str(cache / "bfcl-venv/bin/python"), "bench/bfcl_subset.py",
                     "--manifest", str(manifest_dir / "bfcl-ids.json"), "--split", split,
                     "--output", str(trial / f"bfcl-{tag}"), timeout=1800)
            step(PY, "bench/target_metrics.py", "--output", str(trial / "serving.json"), timeout=480)
            if name != "reference":
                step(PY, "bench/long_context.py", "--near-limit", "--model-path", str(weights),
                     "--output", str(trial / "retrieval.json"), timeout=480)
                step("env", "NUM_PROMPTS=1", "bash", "bench/run_serving_bench.sh",
                     f"continuation-20261119-{name}", timeout=900)

        plan = trial / "plan.json"
        write_json(plan, steps)
        env = environment(model)
        env["HOST"] = "127.0.0.1"
        with (trial / "launcher.log").open("x") as log:
            result = subprocess.run([PY, "bench/mini_probe.py", "--profile", profile,
                                     "--output", str(trial / "probe"), "--validation-plan", str(plan),
                                     "--", *pins, *flags], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        state["profiles"][name] = result.returncode
        write_json(state_path, state)
        if args.phase == "application":
            applications_report(output)
    state.update(status="completed" if all(code == 0 for code in state["profiles"].values())
                 else "completed_with_failures", finished_at=now())
    write_json(state_path, state)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    sys.exit(main())
