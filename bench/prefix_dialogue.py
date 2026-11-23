#!/usr/bin/env python3
"""Measure growing chat histories against identical flushed controls."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from common import ROOT, api, now, write_json
from long_context import FILLER, stream
from probe import capacity


def initial_dialogue(tokenizer, target, rounds, repetition):
    keys = [(f"ARCHIVE_KEY_{repetition}_{i}", f"VALUE_{repetition}_{i}_74309261") for i in range(rounds)]

    def render(count):
        archive, previous = [], 0
        for index, (key, value) in enumerate(keys):
            position = int(count * (.1 + .8 * index / max(1, rounds - 1)))
            archive.extend([FILLER * (position - previous), f"{key} = {value}\n"])
            previous = position
        archive.append(FILLER * (count - previous))
        messages = [{"role": "system", "content": "Retrieve exact archive values. Answer only with the requested value."},
                    {"role": "user", "content": "Read this archive for subsequent questions.\n" +
                     "".join(archive) + f"\nReturn only the value of {keys[0][0]}."}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
        return messages, ids

    count = max(0, (target - 256) // len(tokenizer.encode(FILLER, add_special_tokens=False)))
    messages, ids = render(count)
    while len(ids) > target and count:
        count -= 1
        messages, ids = render(count)
    while True:
        candidate_messages, candidate_ids = render(count + 1)
        if len(candidate_ids) > target:
            break
        count += 1
        messages, ids = candidate_messages, candidate_ids
    if len(ids) > target or target - len(ids) > 64:
        raise ValueError("Could not fit the requested archive budget")
    return messages, keys


def checked_stream(base, ids, expected, limit):
    if len(ids) + 128 > limit:
        raise RuntimeError("Growing dialogue exceeds measured capacity; no silent shortening")
    result = stream(base, ids)
    response = result["response"]
    meta = response["meta_info"]
    output_ids = response.get("output_ids")
    if meta.get("prompt_tokens") != len(ids) or not output_ids or len(output_ids) != meta.get("completion_tokens"):
        raise RuntimeError("Incomplete prompt/output-token evidence")
    cached = meta.get("cached_tokens")
    if not isinstance(cached, int) or not 0 <= cached <= len(ids):
        raise RuntimeError("Missing or invalid cache-token observation")
    result.update(input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        input_tokens=len(ids), cached_tokens=cached,
        retrieval_passed=response.get("text", "").strip() == expected
            and meta.get("finish_reason", {}).get("type") == "stop")
    return result


def gates(dialogues):
    rows = [turn for dialogue in dialogues for turn in dialogue["turns"]]
    followups = [turn for turn in rows if turn["round"] > 0]
    correctness = bool(rows) and all(turn["warm"]["retrieval_passed"] and turn["cold"]["retrieval_passed"] for turn in rows)
    bootstrap = bool(dialogues) and all(dialogue["turns"] and dialogue["turns"][0]["warm"]["cached_tokens"] == 0
                                       for dialogue in dialogues)
    flushed = bool(rows) and all(turn["cold"]["cached_tokens"] == 0 for turn in rows)
    equal = bool(rows) and all(turn["warm"]["response"]["output_ids"] == turn["cold"]["response"]["output_ids"] for turn in rows)
    hits = bool(followups) and all(turn["warm"]["cached_tokens"] > 0 for turn in followups)
    ttft = hits and all(turn["warm"]["ttft_s"] <= 1.5 for turn in followups)
    return {"retrieval_gate": correctness, "cold_bootstrap_gate": bootstrap, "flushed_control_gate": flushed,
            "cache_token_equality_gate": equal, "followup_cache_hit_gate": hits,
            "followup_ttft_1_5s_gate": ttft}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", type=Path, default=ROOT / "weights/target")
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input_tokens < 512 or args.rounds < 2 or args.reps < 1:
        parser.error("Require a new output, at least512 input tokens, two rounds and one repetition")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    info = api(args.base_url, "/get_server_info")
    limit = capacity(info, {})
    result = {"at": now(), "status": "running", "server_info": info,
        "protocol": {"initial_input_budget": args.input_tokens, "rounds": args.rounds,
            "repetitions": args.reps, "temperature": 0, "top_k": 1, "max_new_tokens": 128,
            "thinking": False, "control_order": "full growing conversation, then identical flushed requests"},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dialogues": [],
        "scope": "Archive dialogue and matched cache controls; not general agent task quality"}
    write_json(args.output, result)
    try:
        for repetition in range(args.reps):
            messages, keys = initial_dialogue(tokenizer, args.input_tokens, args.rounds, repetition)
            dialogue = {"repetition": repetition, "initial_messages": list(messages), "turns": []}
            result["dialogues"].append(dialogue)
            api(args.base_url, "/flush_cache", {})
            previous_ids = []
            for index, (key, expected) in enumerate(keys):
                if index:
                    messages.append({"role": "user", "content": f"Return only the value of {key} from the original archive."})
                ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                    add_generation_prompt=True, enable_thinking=False)
                common = 0
                for a, b in zip(previous_ids, ids):
                    if a != b:
                        break
                    common += 1
                warm = checked_stream(args.base_url, ids, expected, limit)
                dialogue["turns"].append({"round": index, "key": key, "expected": expected,
                    "input_ids": ids, "common_prefix_with_previous_input_tokens": common, "warm": warm})
                messages.append({"role": "assistant", "content": warm["response"]["text"]})
                previous_ids = ids
                write_json(args.output, result)
            for turn in dialogue["turns"]:
                api(args.base_url, "/flush_cache", {})
                turn["cold"] = checked_stream(args.base_url, turn["input_ids"], turn["expected"], limit)
                write_json(args.output, result)
        result.update(gates(result["dialogues"]))
        followups = [t for d in result["dialogues"] for t in d["turns"] if t["round"] > 0]
        result["timing"] = {"warm_followup_median_ttft_s": statistics.median(t["warm"]["ttft_s"] for t in followups),
            "warm_followup_max_ttft_s": max(t["warm"]["ttft_s"] for t in followups),
            "identical_flushed_followup_median_ttft_s": statistics.median(t["cold"]["ttft_s"] for t in followups)}
        result["status"] = "passed" if all(result[key] for key in gates(result["dialogues"])) else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result["finished_at"] = now()
        write_json(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
