"""C1 greedy parity against a matched graph control, followed by two timing rounds."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import time

from common import ROOT, api, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output")
    ref = json.loads(args.reference.read_text()) if args.reference else None
    corpus = (ROOT / "bench/prompts_real.jsonl").read_bytes()
    prompts = [json.loads(line)["prompt"] for line in corpus.splitlines()]
    protocol = dict(
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        strict_cap=512,
        timing_cap=256,
        timing_prompts=[0, 3],
        repeats=2,
        return_logprob=False,
    )
    if ref and (ref["status"] != "passed" or ref["protocol"] != protocol):
        raise ValueError("Require a completed control with the same protocol")
    base = "http://127.0.0.1:30000"
    res = dict(
        status="running",
        protocol=protocol,
        server_info=api(base, "/get_server_info"),
        strict=[],
        timing=[],
    )

    def run(idx, fixed):
        api(base, "/flush_cache", {})
        params = {"temperature": 0, "top_k": 1, "max_new_tokens": 512}
        if fixed:
            params.update(max_new_tokens=256, min_new_tokens=256, ignore_eos=True)
        start = time.perf_counter()
        v = api(
            base,
            "/generate",
            {"text": prompts[idx], "sampling_params": params, "return_logprob": False},
        )
        elapsed = time.perf_counter() - start
        ids = v["output_ids"]
        if (
            not ids
            or len(ids) != v["meta_info"]["completion_tokens"]
            or (fixed and len(ids) != 256)
        ):
            raise ValueError("Incomplete generation")
        return dict(prompt=idx, ids=ids, elapsed_s=elapsed, meta=v["meta_info"])

    for idx in (6, 7, 0, 3, 1, 2, 4, 5):
        row = run(idx, False)
        if ref:
            control = next(x for x in ref["strict"] if x["prompt"] == idx)
            row["identical"] = row["ids"] == control["ids"]
            row["first_divergence"] = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(row["ids"], control["ids"]))
                    if a != b
                ),
                None if row["identical"] else min(len(row["ids"]), len(control["ids"])),
            )
        res["strict"].append(row)
        write_json(args.output, res)
        if ref and not row["identical"]:
            res["status"] = "strict_failed"
            write_json(args.output, res)
            return 1
    for rep in range(2):
        rows = [run(idx, True) for idx in protocol["timing_prompts"]]
        res["timing"].append(rows)
        if ref and any(a["ids"] != b["ids"] for a, b in zip(rows, ref["timing"][rep])):
            res["status"] = "timing_parity_failed"
            write_json(args.output, res)
            return 1
        write_json(args.output, res)
    res["mean_round_seconds"] = mean(
        sum(r["elapsed_s"] for r in rows) for rows in res["timing"]
    )
    if ref:
        res["e2e_ratio"] = ref["mean_round_seconds"] / res["mean_round_seconds"]
    res["status"] = "passed"
    write_json(args.output, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
