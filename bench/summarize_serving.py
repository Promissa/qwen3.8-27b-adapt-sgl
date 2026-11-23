#!/usr/bin/env python3
"""Validate the six cookbook-protocol result files and report between-repeat noise."""
import argparse
import json
import math
from pathlib import Path
import statistics
from common import ROOT, now, write_json

METRICS = ("output_throughput", "median_ttft_ms", "median_tpot_ms")


def validate(row, isl):
    if (row.get("backend") != "sglang-oai" or row.get("dataset_name") != "random"
        or row.get("max_concurrency") != 1 or row.get("random_range_ratio") != 1
        or row.get("random_input_len") != isl or row.get("random_output_len") != 1024):
        raise ValueError("Benchmark protocol differs from the required control")
    inputs, outputs = row.get("input_lens", []), row.get("output_lens", [])
    if not outputs or len(inputs) != len(outputs) or row.get("completed") != len(outputs):
        raise ValueError("Missing or incomplete per-request results")
    if len(row.get("errors", [])) != len(outputs) or any(row["errors"]):
        raise ValueError("Requests contain errors or missing error metadata")
    if any(x != isl for x in inputs) or any(x != 1024 for x in outputs):
        raise ValueError("Actual input/output lengths differ from the protocol")
    if row.get("total_input_tokens") != sum(inputs) or row.get("total_output_tokens") != sum(outputs):
        raise ValueError("Aggregate token counts disagree with request details")
    if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) or row[k] <= 0 for k in METRICS):
        raise ValueError("Invalid benchmark timing/throughput")


def summarize(folder):
    result = {"at": now(), "status": "incomplete", "groups": [], "missing": [], "errors": []}
    for isl in (1024, 8192):
        rows = []
        paths = []
        for repetition in (1, 2, 3):
            path = folder / f"isl{isl}-rep{repetition}.jsonl"
            if not path.exists():
                result["missing"].append(str(path))
                continue
            try:
                lines = path.read_text().splitlines()
                if len(lines) != 1:
                    raise ValueError("Expected exactly one result per evidence file")
                row = json.loads(lines[0])
                validate(row, isl)
                rows.append(row)
                paths.append(str(path))
            except (ValueError, TypeError, KeyError) as e:
                result["errors"].append({"path": str(path), "error": str(e)})
        group = {"input_tokens": isl, "output_tokens": 1024, "repetitions": len(rows), "evidence": paths}
        if len(rows) == 3:
            group["metrics"] = {}
            for metric in METRICS:
                values = [r[metric] for r in rows]
                mean = statistics.mean(values)
                group["metrics"][metric] = {"values": values, "mean": mean,
                    "median": statistics.median(values), "min": min(values), "max": max(values),
                    "sample_stddev": statistics.stdev(values),
                    "coefficient_of_variation": statistics.stdev(values) / mean}
        result["groups"].append(group)
    if not result["missing"] and not result["errors"]:
        result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args()
    folder = ROOT / "results/bench" / args.tag
    report = summarize(folder)
    write_json(folder / "summary.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
