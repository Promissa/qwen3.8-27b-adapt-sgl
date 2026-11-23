#!/usr/bin/env python3
"""Exact-key retrieval at three depths, plus measured prefix reuse and TTFT.

This is a functional quality gate, separate from synthetic token capacity tests.
The tokenizer is local; generation uses the same explicit chat-template IDs for
all compared recipes. All raw responses and observed cache counts are retained.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

from common import ROOT, api, fields, now, write_json

FILLER = "Archive entry: routine maintenance completed without an exceptional event.\n"


def make_prompt(tokenizer, target, depth, key):
    def render(count):
        before = int(count * depth)
        content = ("Read the archive and retrieve the exact value of the UNIQUE_ARCHIVE_KEY.\n" +
            FILLER * before + f"UNIQUE_ARCHIVE_KEY = {key}\n" + FILLER * (count - before) +
            "\nReturn only the value of UNIQUE_ARCHIVE_KEY, without explanation.")
        return tokenizer.apply_chat_template([{"role": "user", "content": content}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    count = max(0, (target - 128) // len(tokenizer.encode(FILLER, add_special_tokens=False)))
    ids = render(count)
    while len(ids) > target and count > 0:
        count -= 1
        ids = render(count)
    while True:
        candidate = render(count + 1)
        if len(candidate) > target:
            break
        count += 1
        ids = candidate
    if len(ids) > target or target - len(ids) > 64:
        raise RuntimeError(f"Could not construct requested prompt length: {len(ids)} vs {target}")
    return ids


def stream(base, ids):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": 128}}
    req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    last = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if first is None and event.get("text"):
                first = time.perf_counter() - start
            last = event
    if not last or first is None:
        raise RuntimeError("No streamed output received")
    return {"ttft_s": first, "elapsed_s": time.perf_counter() - start, "response": last}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", type=Path, default=ROOT / "weights/target")
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument("--input-tokens", type=int)
    length.add_argument("--near-limit", action="store_true",
                        help="Use the reported live capacity minus the multi-turn reserve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.input_tokens is not None and args.input_tokens < 512):
        parser.error("Require a new output file and at least 512 prompt tokens")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    result = {"at": now(), "status": "running", "input_tokens_requested": args.input_tokens,
              "length_mode": "near_limit" if args.near_limit else "fixed",
              "server_info": api(args.base_url, "/get_server_info"), "checks": []}
    try:
        limits = [x for name in ("max_total_num_tokens", "context_length")
                  for x in fields(result["server_info"], name) if isinstance(x, int) and x > 0]
        if limits:
            result["reported_capacity"] = min(limits)
            if args.near_limit:
                args.input_tokens = min(limits) - 512
                result["resolved_input_tokens"] = args.input_tokens
            # Reserve both answer generations and the appended user turn. Fail
            # explicitly instead of silently shortening a requested quality test
            # or leaving an impossible request queued indefinitely.
            if args.input_tokens + 512 > min(limits):
                raise RuntimeError(f"Requested {args.input_tokens} input tokens plus multi-turn reserve "
                                   f"exceeds reported capacity {min(limits)}")
        elif args.near_limit:
            raise RuntimeError("Cannot derive a near-limit input without reported capacity")
        if args.input_tokens < 512:
            raise RuntimeError("Insufficient capacity for a multi-turn retrieval test")
        for index, depth in enumerate((.1, .5, .9)):
            # Deterministic keys keep baseline/candidate inputs identical.
            key = f"ARCHIVE_{args.input_tokens}_{index}_74309261"
            ids = make_prompt(tokenizer, args.input_tokens, depth, key)
            api(args.base_url, "/flush_cache", {})
            cold = stream(args.base_url, ids)
            # Preserve the exact cached prefix and generated answer, then add a
            # correctly templated user turn. Do not append a question inside the
            # assistant's unfinished generation prompt.
            marker = "ASSISTANT_HISTORY_PLACEHOLDER"
            followup = tokenizer.apply_chat_template([
                {"role": "assistant", "content": marker},
                {"role": "user", "content": "Repeat only the archive key value."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            suffix = tokenizer.encode(followup.split(marker, 1)[1], add_special_tokens=False)
            generated = list(cold["response"].get("output_ids", []))
            if not generated:
                raise RuntimeError("Missing output IDs needed for exact-prefix multi-turn replay")
            if generated[-1] == tokenizer.eos_token_id:
                generated.pop()
            warm = stream(args.base_url, ids + generated + suffix)
            cold_text = cold["response"].get("text", "").strip()
            warm_text = warm["response"].get("text", "").strip()
            cached = warm["response"].get("meta_info", {}).get("cached_tokens")
            row = {"depth": depth, "expected": key, "input_tokens": len(ids),
                   "cold": cold, "warm": warm, "retrieval_passed": cold_text == key and warm_text == key,
                   "prefix_hit_observed": isinstance(cached, int) and cached > 0,
                   "prefix_hit_ttft_target_passed": isinstance(cached, int) and cached > 0 and warm["ttft_s"] <= 1.5}
            result["checks"].append(row)
            write_json(args.output, result)
        result["status"] = "passed" if all(x["retrieval_passed"] for x in result["checks"]) else "failed"
        result["prefix_hit_ttft_gate"] = all(x["prefix_hit_ttft_target_passed"] for x in result["checks"])
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        write_json(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
