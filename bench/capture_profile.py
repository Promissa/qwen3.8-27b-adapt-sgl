#!/usr/bin/env python3
"""Capture CPU/GPU/MEM trace using SGLang's stage-aware profiler."""
import argparse
import gzip
import json
from pathlib import Path
import uuid
from common import ROOT, api, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"at": now(), "status": "running", "server_info": api(args.base_url, "/get_server_info")}
    prompts = ["Explain the difference between a process and a thread."] * 3
    for prompt in prompts:  # warmup, outside trace
        api(args.base_url, "/generate", {"text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}})
    api(args.base_url, "/start_profile", {"output_dir": str(args.output.resolve()),
        "activities": ["CPU", "GPU", "MEM"], "profile_by_stage": True,
        # The legacy stage profiler initializes its counters only with num_steps.
        "num_steps": 32, "profile_id": uuid.uuid4().hex,
        "with_stack": True, "record_shapes": True})
    try:
        result["response"] = api(args.base_url, "/generate", {
            "text": "Write a Python LRU cache with tests and explain the design.",
            "sampling_params": {"temperature": 0, "max_new_tokens": 256,
                                "min_new_tokens": 256, "ignore_eos": True}})
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            api(args.base_url, "/stop_profile", {})
        except Exception as cleanup:
            result["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
        raise
    finally:
        # A completed 256-token request has already auto-stopped the bounded
        # 32-step profiler. A second stop is rejected with HTTP 500 by this commit.
        write_json(args.output / "capture.json", result)
    traces = sorted(args.output.glob("*.trace.json.gz"))
    result["traces"] = []
    for trace in traces:
        with gzip.open(trace, "rt") as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        result["traces"].append({"path": str(trace), "gpu_kernel_events": sum(
            e.get("cat") == "kernel" for e in events)})
    if not any(t["gpu_kernel_events"] > 0 and "DECODE" in t["path"] for t in result["traces"]):
        result["status"] = "failed"
        write_json(args.output / "capture.json", result)
        raise RuntimeError("No decode GPU-kernel trace was produced")
    result["status"] = "passed"
    write_json(args.output / "capture.json", result)


if __name__ == "__main__":
    main()
