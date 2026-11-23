"""Two matched cold and cached Balanced requests per condition, with stage timing."""

import argparse
import random
from pathlib import Path
from statistics import mean

from common import ROOT, api, write_json
from long_context import make_prompt, stream
from target_metrics import request


def stages(meta):
    return {
        "forward_ms": 1000
        * (meta["prefill_finished_time"] - meta["forward_entry_time"]),
        "queue_ms": 1000 * meta["queue_time"],
        "cached": meta["cached_tokens"],
        "input": meta["prompt_tokens"],
        "server_first_ms": 1000 * meta["first_token_latency"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url
    info = api(base, "/get_server_info")
    result = {"server_info": info, "cold": [], "warm": [], "status": "running"}
    dest = args.output / "screen.json"
    write_json(dest, result)
    # Same complete 8K prompt shape warmup after each model boot, outside measurement.
    for rep in (-1, 0, 1):
        rng = random.Random(9893 + rep)
        ids = [rng.randrange(1000, 20000) for _ in range(8192)]
        api(base, "/flush_cache", {})
        row = request(
            base,
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "top_k": 1,
                    "min_new_tokens": 4,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            },
            args.output / f"cold-{rep}.jsonl",
        )
        row.update(stages(row["final_meta"]))
        if rep >= 0:
            result["cold"].append(row)
            write_json(dest, result)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        ROOT / "weights/target-embedding-nvfp4", local_files_only=True
    )
    marker = "ASSISTANT_HISTORY_PLACEHOLDER"
    template = tok.apply_chat_template(
        [
            {"role": "assistant", "content": marker},
            {"role": "user", "content": "Repeat only the archive key value."},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    suffix = tok.encode(template.split(marker, 1)[1], add_special_tokens=False)
    cases = []
    for i, depth in enumerate((0.1, 0.5, 0.9)):
        key = f"ARCHIVE_48640_{i}_74309261"
        cases.append((depth, key, make_prompt(tok, 48640, depth, key)))
    for rep, order in enumerate(((0, 1, 2), (2, 1, 0))):
        for idx in order:
            depth, key, ids = cases[idx]
            api(base, "/flush_cache", {})
            cold = stream(base, ids)
            out = list(cold["response"]["output_ids"])
            if out and out[-1] == tok.eos_token_id:
                out.pop()
            warm = stream(base, ids + out + suffix)
            row = dict(
                rep=rep,
                depth=depth,
                cold=cold,
                warm=warm,
                ttft_ms=warm["ttft_s"] * 1000,
                **stages(warm["response"]["meta_info"]),
            )
            row["passed"] = (
                len(ids) == 48640
                and row["cached"] == 48640
                and all(r["response"]["text"].strip() == key for r in (cold, warm))
            )
            result["warm"].append(row)
            write_json(dest, result)
    result["cold_mean_ms"] = mean(r["ttft_ms"] for r in result["cold"])
    result["warm_means_ms"] = {
        str(depth): mean(r["ttft_ms"] for r in result["warm"] if r["depth"] == depth)
        for depth, _, _ in cases
    }
    result["status"] = (
        "passed" if all(r["passed"] for r in result["warm"]) else "retrieval_failed"
    )
    write_json(dest, result)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
