#!/usr/bin/env python3
"""Summarize the continuation from raw artifacts, including incomplete phases."""
import argparse
import json
from pathlib import Path
import time

from continue_targets import now, write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def profile_report(root, profile):
    trial = root / "application" / profile
    bfcl = {}
    for split in ("screen", "holdout"):
        scored = read(trial / f"bfcl-{split}/summary.json")
        if scored.get("status") == "failed":
            scored = read(trial / f"bfcl-{split}-rescore/summary.json") or scored
        bfcl[split] = {key: scored[key] for key in ("status", "passed", "total", "capacity_rejected")
                       if key in scored}

    longbench = {}
    for split in ("screen", "holdout"):
        result = read(trial / f"longbench-{split}.json")
        if result:
            longbench[split] = {key: result[key] for key in ("status", "passed", "total") if key in result}
            longbench[split]["capped"] = sum(
                row["response"]["meta_info"]["finish_reason"]["type"] == "length"
                for row in result["results"])

    retrieval = read(trial / "retrieval.json")
    serving = read(trial / "serving.json")
    soak = read(root / "soak" / profile / "soak/soak.json")
    return {
        "bfcl": bfcl,
        "longbench": longbench,
        "serving_status": serving.get("status"),
        "serving_metrics": serving.get("summary"),
        "retrieval_status": retrieval.get("status"),
        "retrieval_tokens": [row["input_tokens"] for row in retrieval.get("checks", [])],
        "warm_ttft_ms": [row["warm"]["ttft_s"] * 1000 for row in retrieval.get("checks", [])],
        "warm_cache_hits": [row["prefix_hit_observed"] for row in retrieval.get("checks", [])],
        "soak_status": ("cancelled_by_user" if soak else "not_started")
                       if (root / "soak/cancellation.json").exists() else soak.get("status"),
        "soak_elapsed_s": soak.get("elapsed_s"),
        "soak_7200_s_passed": soak.get("runtime_stability_7200s_gate"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()
    root = args.directory
    while True:
        active = False
        if args.watch_pid:
            cmdline = Path(f"/proc/{args.watch_pid}/cmdline")
            try:
                active = b"run_remaining_tests.sh" in cmdline.read_bytes()
            except FileNotFoundError:
                pass
        progress = {
            "updated_at": now(),
            "queue_active": active,
            "phases": {name: read(root / name / "runs.json").get("status")
                       for name in ("memory", "spec", "application", "matched-serving", "soak")},
            "profiles": {profile: profile_report(root, profile)
                         for profile in ("reference", "balanced", "max-context")},
            "paired_quality": read(root / "application/paired-quality.json"),
        }
        write_json(root / "progress.json", progress)
        if not active:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
