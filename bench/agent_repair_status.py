"""Summarize official trial results for the reviewed 2026-11-19 benchmark subset."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def framework_trial(folder):
    # The job-level result can exist while a trial is still running.
    for path in sorted(folder.glob("*/result.json")):
        result = read(path)
        if not result.get("finished_at"):
            continue
        verifier = result.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        return {
            "finished_at": result["finished_at"],
            "reward": rewards.get("reward"),
            "rewards": rewards,
            "exception": (result.get("exception_info") or {}).get("exception_type"),
            "evidence": str(path.relative_to(folder.parent.parent)),
        }
    return None


def terminal_trial(root, job):
    original = framework_trial(root / "harbor" / job)
    if original and original["reward"] is not None:
        return original
    retry = framework_trial(root / "harbor" / job.replace("-model", "-tools-model"))
    if retry:
        retry["harness_tool_preinstalled"] = True
        if original:
            retry["initial_setup_error"] = original["evidence"]
        return retry
    return original


def ale_trial(root, experiment):
    folder = root / "ale" / experiment
    attempts = sorted(folder.glob("*/*/*/v0/[0-9]*"))
    if not attempts:
        return {"state": "queued" if experiment.endswith("-native") else "waiting_for_guest"}
    attempt = attempts[-1]
    path = attempt / "eval_result.json"
    result = read(path)
    run = read(attempt / "run.json")
    if result is None or run is None:
        return {"state": "awaiting_result", "evidence": str(attempt.relative_to(root))}
    valid = result["eval_status"] == "success" and run["status"] in ("completed", "success", "timeout")
    return {
        "state": "graded" if valid and result["score"] is not None else "error",
        "reward": result["score"],
        "eval_status": result["eval_status"],
        "run_status": run["status"],
        "seconds": (run.get("timings") or {}).get("duration_s"),
        "evidence": str(path.relative_to(root)),
    }


def summarize(root):
    records = []
    selected = read(root / "selected.json")
    terminal_jobs = {"build-pmars": "pmars-model", "build-cython-ext": "cython-model",
                     "polyglot-c-py": "polyglot-c-py-model", "query-optimize": "query-optimize-model"}
    cases = {"abs-stepped-slices": "slices", "tomlkit-toml-table-converters": "tomlkit",
             "sqlfmt-create-table-ddl-formatting": "sqlfmt"}
    pro_jobs = {"96b99780": "duration", "a84ecfb8": "suggest", "ff1c025a": "fonts"}
    for task in selected:
        suite = task["suite"]
        row = {"suite": suite, "instance": task["instance"], "state": "pending"}
        if suite == "terminal":
            job = terminal_jobs[task["instance"]]
            result = terminal_trial(root, job)
            if result:
                row.update(result, state="graded" if result["reward"] is not None else "error")
            elif (root / "harbor" / job).is_dir():
                row["state"] = "running"
        elif suite in ("swe", "deep"):
            case = (next(value for commit, value in pro_jobs.items() if commit in task["instance"])
                    if suite == "swe" else cases[task["instance"]])
            meta = read(root / ("qwen-" + case) / "metadata.json")
            if meta:
                row["actor"] = meta
                row["state"] = "awaiting_verifier"
            elif (root / ("qwen-" + case)).is_dir():
                row["state"] = "running"
            result = None
            if suite == "swe":
                dest = "swe-qwen" if case == "duration" else "swe-" + case + "-qwen"
                path = root / dest / "eval_results.json"
                evaluation = read(path) or {}
                if task["instance"] in evaluation:
                    result = {"reward": int(evaluation[task["instance"]]), "evidence": str(path.relative_to(root))}
            else:
                result = framework_trial(root / "pier" / (case + "-qwen"))
            if result:
                row.update(result)
                row["state"] = "graded" if result["reward"] is not None else "error"
                if not meta or meta["exit_code"] not in (0, 124) or meta.get("is_error"):
                    row["state"] = "actor_error"
        elif suite == "ale":
            row.update(ale_trial(root, "qwen-local-v1-office"))
        if suite == "terminal":
            replay = root / (task["instance"] + "-recovered") / "ctrf.json"
            if replay.exists():
                counts = read(replay)["results"]["summary"]
                row["artifact_replay"] = {"passed": counts["passed"], "tests": counts["tests"],
                                          "evidence": str(replay.relative_to(root))}
        records.append(row)
    totals = {"terminal": 89, "swe": 731, "deep": 113, "ale": 152}
    coverage = {}
    for suite, total in totals.items():
        picked = [item for item in records if item["suite"] == suite]
        graded = [item for item in picked if item["state"] == "graded"]
        coverage[suite] = dict(selected=len(picked), graded=len(graded), release_total=total,
                               passed=sum(item["reward"] == 1 for item in graded))
    return {"coverage": coverage, "tasks": records,
            "cython_environment_corrected": terminal_trial(root, "cython-pinned-model"),
            "ale_native_tools": ale_trial(root, "qwen-local-v1-office-native")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.results), indent=2, ensure_ascii=False))
