#!/usr/bin/env python3
"""Attribute the measured Qwen3.8-27B baseline decode trace, with overlap explicit."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
from common import now, write_json


def analyze(path, grid_layout="baseline"):
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    events = trace["traceEvents"]
    steps = [e for e in events if e.get("cat") == "user_annotation" and e["name"] == "step[DECODE bs=1]"]
    if not steps:
        raise ValueError("No single-stream decode steps in trace")
    kernels = [e for e in events if e.get("cat") == "kernel"]
    gemms = sorted((e for e in kernels if "cutlass13device_kernel" in e["name"]), key=lambda e: e["ts"])
    if len(gemms) != len(steps) * 257:
        raise ValueError("Trace does not match the baseline's 64-layer, 257-GEMM decode path")
    attribution = {}
    for step in range(len(steps)):
        for layer in range(64):
            group = gemms[step * 257 + layer * 4:step * 257 + layer * 4 + 4]
            full = layer % 4 == 3
            expected = [(1, 112, 1) if full else (128, 1, 1), (1, 40, 1), (1, 170, 1), (1, 40, 1)]
            if grid_layout == "balanced-20261119":
                # Current autotuned CUTLASS kernels: the fused gate/up projects to
                # 2*17408 columns (272 tiles of 128), down projects to 5120.
                expected = [(1, 112 if full else 128, 1), (40, 1, 3), (272, 1, 1), (1, 40, 1)]
            if [tuple(e["args"]["grid"]) for e in group] != expected:
                raise ValueError(f"Unverified layer GEMM order at step={step}, layer={layer}")
            for index, event in enumerate(group):
                attribution[id(event)] = ("mlp_gemm" if index >= 2 else
                    "attention_projection_gemm" if full else "gdn_projection_gemm")
        head = gemms[step * 257 + 256]
        if tuple(head["args"]["grid"]) != (170, 1, 1):
            raise ValueError("Unverified lm_head kernel")
        attribution[id(head)] = "lm_head_gemm"

    def category(event):
        if id(event) in attribution:
            return attribution[id(event)]
        name = event["name"]
        if "gemvx::kernel" in name:
            return "gdn_ba_projection_gemv"
        if any(x in name for x in ("gated_delta_rule", "qkvzba_causal_conv", "_layer_norm_fwd_1pass")):
            return "gdn_recurrence_conv_norm"
        if any(x in name for x in ("PagedKVCacheKernel", "MergeStatesKernel", "qk_rmsnorm_rope", "store_kvcache", "_fused_sigmoid_mul")):
            return "attention_core_rope_kv"
        if "quantize_with_block_size" in name or "cvt_fp16_to_fp4_expert" in name:
            return "fp4_activation_quantization"
        if "rmsnorm" in name.lower() or "rms_norm" in name.lower():
            return "residual_norm"
        if "argmax" in name.lower() or "sampling" in name.lower():
            return "sampling"
        return "other_pointwise_and_bookkeeping"

    by_category = defaultdict(lambda: {"kernel_calls": 0, "summed_us": 0.})
    by_name = defaultdict(lambda: {"kernel_calls": 0, "summed_us": 0.})
    for event in kernels:
        for stats in (by_category[category(event)], by_name[event["name"]]):
            stats["kernel_calls"] += 1
            stats["summed_us"] += event["dur"]
    total = sum(e["dur"] for e in kernels)
    for stats in by_category.values():
        stats["summed_ms_per_decode_step"] = stats["summed_us"] / len(steps) / 1000
        stats["share_of_summed_kernel_time"] = stats["summed_us"] / total
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    union = []
    for begin, end in intervals:
        if union and begin <= union[-1][1]:
            union[-1] = (union[-1][0], max(union[-1][1], end))
        else:
            union.append((begin, end))
    host = defaultdict(lambda: {"calls": 0, "inclusive_us": 0.})
    for e in events:
        if e.get("cat") == "user_annotation" or (e.get("cat") == "cuda_runtime" and "Synchronize" in e["name"]):
            host[e["name"]]["calls"] += 1
            host[e["name"]]["inclusive_us"] += e.get("dur", 0)
    for stats in host.values():
        stats["inclusive_ms_per_decode_step"] = stats["inclusive_us"] / len(steps) / 1000
    result = {"at": now(), "trace": str(path), "decode_steps": len(steps),
        "gemm_order_verified": True, "grid_layout": grid_layout, "gemm_calls_per_step": 257,
        "gpu_kernel_summed_ms": total / 1000,
        "gpu_kernel_union_ms": sum(end - begin for begin, end in union) / 1000,
        "gpu_kernel_span_ms": (max(end for _, end in intervals) - intervals[0][0]) / 1000,
        "gpu_categories": dict(by_category), "host_inclusive_annotations": dict(host),
        "top_kernels": sorted(({"name": k, **v} for k, v in by_name.items()), key=lambda x: x["summed_us"], reverse=True)[:25],
        "limitations": ["Kernel durations overlap across streams; category sums are not wall-time percentages.",
            "Host annotations include GPU synchronization and overlap with kernels; they are not CPU compute costs.",
            "This source-verified GEMM grid/order mapping is specific to the measured CUTLASS baseline.",
            "Profiling perturbs timing; use the independent serving benchmark for throughput and TPOT."]}
    snapshots = list(path.parent.glob("*memory-DECODE.pickle"))
    if snapshots:
        # These are the trusted snapshots produced by this workspace's profiler.
        with snapshots[0].open("rb") as stream:
            snapshot = pickle.load(stream)
        segments = snapshot["segments"]
        result["allocator_snapshot"] = {"path": str(snapshots[0]), "segments": len(segments),
            "reserved_bytes": sum(s["total_size"] for s in segments),
            "allocated_bytes": sum(s["allocated_size"] for s in segments),
            "active_bytes_including_pending_free": sum(s["active_size"] for s in segments)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-layout", choices=("baseline", "balanced-20261119"), default="baseline")
    args = parser.parse_args()
    write_json(args.output, analyze(args.trace, args.grid_layout))
