#!/usr/bin/env python3
"""Verify full, matched GSM8K evidence before applying PLAN.md's quality gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from common import now, write_json


def read_run(directory):
    directory = Path(directory)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "output-rs0.jsonl"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("name") != "gsm8k" or metrics.get("num_examples") != 1319 or metrics.get("n_repeats") != 1:
        raise ValueError("Require a completed 1319-example, one-repeat GSM8K run")
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in predictions]
    if len(ids) != 1319 or set(ids) != {f"gsm8k-{i}" for i in range(1319)}:
        raise ValueError("Missing, duplicate or unexpected prediction IDs")
    protocol = metrics["generation"]
    cap = protocol.get("max_tokens")
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError("Output budget must be explicit and positive")
    for row in predictions:
        if not isinstance(row.get("symbolic_correct"), bool):
            raise ValueError(f"Missing official symbolic score: {row['id']}")
        if row.get("finish_reason") not in ("stop", "length"):
            raise ValueError(f"Unsuccessful request: {row['id']}")
        if not isinstance(row.get("num_generated_tokens"), int) or row["num_generated_tokens"] <= 0:
            raise ValueError(f"Missing generated-token evidence: {row['id']}")
        if row["num_generated_tokens"] > cap or (row["finish_reason"] == "length"
                and row["num_generated_tokens"] != cap):
            raise ValueError(f"Actual output budget differs from requested cap: {row['id']}")
    correct = sum(row["symbolic_correct"] for row in predictions)
    if not math.isclose(metrics["aggregate"]["score"], correct / 1319, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Aggregate score disagrees with official prediction rows")
    if metrics["aggregate"].get("error_rate", 0) != 0:
        raise ValueError("Official evaluator recorded request errors")
    return {"directory": str(directory.resolve()), "correct": correct, "n": 1319,
        "accuracy_percent": 100 * correct / 1319,
        "truncated": sum(row["finish_reason"] == "length" for row in predictions),
        "generated_tokens": sum(row["num_generated_tokens"] for row in predictions),
        "protocol": protocol,
        "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (metrics_path, predictions_path)},
        "rows": {row["id"]: row for row in predictions}}


def compare(baseline_dir, candidate_dir, max_drop_pp=.5):
    if not math.isfinite(max_drop_pp) or max_drop_pp < 0:
        raise ValueError("Quality tolerance must be finite and nonnegative")
    baseline, candidate = read_run(baseline_dir), read_run(candidate_dir)
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("Different generation protocols; not a matched weight comparison")
    lost, gained = [], []
    for identifier, row in baseline["rows"].items():
        other = candidate["rows"][identifier]
        if (row["problem"], row["expected_answer"]) != (other["problem"], other["expected_answer"]):
            raise ValueError(f"Different question or expected answer: {identifier}")
        if row["symbolic_correct"] and not other["symbolic_correct"]:
            lost.append(identifier)
        if other["symbolic_correct"] and not row["symbolic_correct"]:
            gained.append(identifier)
    drop_pp = 100 * (baseline["correct"] - candidate["correct"]) / 1319
    del baseline["rows"], candidate["rows"]
    return {"at": now(), "baseline": baseline, "candidate": candidate,
        "accuracy_drop_percentage_points": drop_pp, "max_drop_percentage_points": max_drop_pp,
        "gsm8k_gate_passed": drop_pp <= max_drop_pp,
        "lost_correct_ids": lost, "gained_correct_ids": gained,
        "full_plan_promotion_verified": False,
        "scope": "Full matched GSM8K gate only; auxiliary quality, retrieval, speed and stability remain separate"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence file")
    result = compare(args.baseline_dir, args.candidate_dir)
    write_json(args.output, result)
    raise SystemExit(0 if result["gsm8k_gate_passed"] else 1)
