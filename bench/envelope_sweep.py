#!/usr/bin/env python3
"""Probe endpoints, then coarse discovery/bisection if the upper endpoint fails."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

from common import ROOT, now, write_json


def classify(result, log_text):
    if result.get("status") == "passed":
        return "passed"
    if "Loaded weights leave no GPU memory for the KV cache under" in log_text:
        return "static_budget_failure"
    if any(term in log_text.lower() for term in (
            "out of memory", "insufficient memory", "not enough memory")):
        return "memory_failure"
    error = result.get("error") or ""
    if "Concurrent capacity gate did not observe " in error:
        return "concurrency_not_demonstrated"
    if "Reported capacity " in error and "cannot fit 8192/1024" in error:
        return "insufficient_token_pool"
    # Merely reporting a pool does not turn a later protocol/client failure
    # into a negative memory measurement.
    return "infrastructure_or_unclassified_failure"


def bracket_search(test, low, high, precision, step=.02, prior_samples=None):
    samples = dict(prior_samples or {})

    def check(x):
        x = round(x, 5)
        if x not in samples:
            samples[x] = bool(test(x))
        return samples[x]

    check(low)  # Retain a conservative operating-point/time observation.
    if check(high):
        return round(high, 5), samples
    # Both endpoints may fail while a middle interval passes. Preserve coarse
    # discovery whenever the upper endpoint does not settle the bounded search.
    points = [low + i * step for i in range(int((high - low) / step) + 1)] + [high]
    for point in points:
        check(min(point, high))
    passed = [x for x, ok in samples.items() if ok]
    if not passed:
        return None, samples
    left = max(passed)
    right = min([x for x, ok in samples.items() if x > left and not ok], default=high)
    while right - left > precision:
        mid = round((left + right) / 2, 5)
        if check(mid):
            left = mid
        else:
            right = mid
    return left, samples


def load_progress(path, desktop, labels, low, high):
    """Reuse completed, classified measurements; never hide a failed protocol."""
    summary_path = path.with_suffix('.summary.json')
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
    if len({entry['combination'] for entry in summaries}) != len(summaries):
        raise ValueError('Duplicate completed combination')
    measured = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            label = f"{row['desktop']}-{row['strategy']}-{int(row['prefill_chunk'])}-c{int(row['concurrency'])}-{row['dtype']}"
            fraction = float(row['mem_fraction'])
            if row['desktop'] != desktop or label not in labels or not low <= fraction <= high:
                raise ValueError('Resume protocol differs from retained measurements')
            if row['outcome'] == 'infrastructure_or_unclassified_failure':
                raise ValueError('Inspect retained infrastructure failure before resuming')
            passed = row['status'] == 'passed'
            if passed != (row['outcome'] == 'passed'):
                raise ValueError('Retained status/classification mismatch')
            samples = measured.setdefault(label, {})
            if fraction in samples:
                raise ValueError('Duplicate measured fraction in combination')
            samples[fraction] = passed
    if not measured:
        raise ValueError('No retained measurements to resume')
    for entry in summaries:
        label = entry['combination']
        samples = measured.get(label, {})
        expected = {float(key): value for key, value in entry['samples'].items()}
        passed = [fraction for fraction, ok in samples.items() if ok]
        if (label not in labels or expected != samples or
                entry['highest_tested_passing_fraction'] != (max(passed) if passed else None)):
            raise ValueError('Completed search summary disagrees with retained CSV')
    return summaries, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", choices=["present", "absent"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/envelope.csv")
    parser.add_argument("--low", type=float, default=.88)
    parser.add_argument("--high", type=float, default=.98)
    parser.add_argument("--precision", type=float, default=.005)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse the same retained CSV and completed search summaries")
    args = parser.parse_args()
    if not (0 < args.low <= args.high < 1) or args.precision <= 0:
        parser.error("Require 0 < low <= high < 1 and positive precision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = now().replace(":", "").replace("+", "_")
    combinations = list(itertools.product(
        ["extra_buffer", "extra_buffer_lazy", "no_buffer", "disabled"],
        [2048, 1024, 512], args.concurrency))
    # One fp32 feasibility control, not a full fp32 grid.
    grid = [(strategy, chunk, conc, "bfloat16") for strategy, chunk, conc in combinations]
    grid += [("extra_buffer_lazy", 2048, 1, "float32")]
    labels = {f'{args.desktop}-{strategy}-{chunk}-c{conc}-{dtype}' for strategy, chunk, conc, dtype in grid}
    if args.resume and not args.dry_run:
        summaries, measured = load_progress(args.output, args.desktop, labels, args.low, args.high)
    else:
        if args.output.exists() and not args.dry_run:
            parser.error('Existing evidence requires explicit --resume')
        summaries, measured = [], {}
    completed = {entry['combination'] for entry in summaries}
    for strategy, chunk, conc, dtype in grid:
        label = f"{args.desktop}-{strategy}-{chunk}-c{conc}-{dtype}"
        if label in completed:
            continue

        def test(fraction):
            folder = ROOT / "results/probes" / run_id / f"{label}-{fraction:.5f}"
            cmd = [sys.executable, str(ROOT / "bench/probe.py"), "--output", str(folder),
                   "--desktop", args.desktop, "--concurrency", str(conc), "--",
                   "--mem-fraction-static", str(fraction), "--chunked-prefill-size", str(chunk),
                   "--mamba-ssm-dtype", dtype]
            cmd += (["--disable-radix-cache"] if strategy == "disabled" else
                    ["--mamba-radix-cache-strategy", strategy])
            if strategy == "no_buffer":
                # Required by upstream validation; this policy necessarily
                # changes scheduling as well as state-buffer ownership.
                cmd += ["--disable-overlap-schedule"]
            if args.dry_run:
                print(json.dumps(cmd))
                return False
            subprocess.run(cmd, cwd=ROOT, check=False)
            result_file = folder / "probe.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {"status": "driver_failed"}
            log_path = folder / "server.log"
            log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
            outcome = classify(result, log_text)
            row = {"at": now(), "desktop": args.desktop, "strategy": strategy, "dtype": dtype,
                   "prefill_chunk": chunk, "concurrency": conc, "mem_fraction": fraction,
                   "status": result["status"], "outcome": outcome, "capacity": result.get("reported_context_capacity"),
                   "peak_device_used_mib": result.get("peak_device_used_mib"), "evidence": str(result_file)}
            has_header = args.output.exists() and args.output.stat().st_size > 0
            with args.output.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row)
                if not has_header:
                    writer.writeheader()
                writer.writerow(row)
            # Infrastructure/download failures are not negative memory measurements.
            if outcome == "infrastructure_or_unclassified_failure":
                raise RuntimeError(f"Infrastructure/startup failure; inspect {result_file}")
            return result["status"] == "passed"

        if args.dry_run:
            test(args.low)
            continue
        best, samples = bracket_search(test, args.low, args.high, args.precision,
                                      prior_samples=measured.get(label))
        summaries.append({"combination": label, "highest_tested_passing_fraction": best,
                          "samples": samples, "global_maximum_proven": False,
                          "search_protocol": "low_high_then_coarse_if_upper_fails",
                          "retained_sample_count": len(measured.get(label, {})),
                          "new_sample_count": len(samples) - len(measured.get(label, {}))})
        write_json(args.output.with_suffix(".summary.json"), summaries)


if __name__ == "__main__":
    main()
