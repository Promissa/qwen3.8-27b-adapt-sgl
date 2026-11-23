#!/usr/bin/env python3
"""Capture greedy token IDs on real prompts, compare across sequential server boots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from common import ROOT, api, now, write_json


def compare(reference, candidate):
    if not reference.get("complete") or not candidate.get("complete"):
        raise ValueError("Incomplete token capture cannot establish equality")
    if reference["protocol"] != candidate["protocol"]:
        raise ValueError("Different prompts or sampling protocol")
    expected = reference["protocol"].get("prompt_count", 0)
    if expected < 1 or len(reference["outputs"]) != expected or len(candidate["outputs"]) != expected:
        return False
    return all(a["output_ids"] == b["output_ids"]
               for a, b in zip(reference["outputs"], candidate["outputs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompts", type=Path, default=ROOT / "bench/prompts_real.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    corpus = args.prompts.read_bytes()
    if not corpus.strip():
        parser.error("Prompt corpus must not be empty")
    protocol = {"corpus_sha256": hashlib.sha256(corpus).hexdigest(),
                "prompt_count": len(corpus.decode().splitlines()),
                "sampling_params": {"temperature": 0, "top_k": 1, "max_new_tokens": 512}}
    outputs = []
    result = {"at": now(), "protocol": protocol, "complete": False,
              "server_info": api(args.base_url, "/get_server_info"), "outputs": outputs}
    for row in corpus.decode().splitlines():
        prompt = json.loads(row)["prompt"]
        response = api(args.base_url, "/generate", {"text": prompt,
            "sampling_params": protocol["sampling_params"], "return_logprob": True})
        ids = response.get("output_ids")
        if ids is None:
            ids = [entry[1] for entry in response.get("meta_info", {}).get("output_token_logprobs", [])]
        if not ids or len(ids) != response["meta_info"]["completion_tokens"]:
            raise RuntimeError("Missing complete output token IDs; text equality cannot prove token equality")
        outputs.append({"output_ids": ids, "text": response.get("text"), "meta_info": response["meta_info"]})
        write_json(args.output, result)
    result["complete"] = True
    write_json(args.output, result)
    if args.reference:
        result["token_identical"] = compare(json.loads(args.reference.read_text()), result)
        write_json(args.output, result)
        return 0 if result["token_identical"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
