#!/usr/bin/env python3
"""Freeze and run the project's LongBench v2 zero-shot subset."""
import argparse
import json
from pathlib import Path
import random
import re
import time

from common import api, now, write_json

DOMAINS = ("Single-Document QA", "Code Repository Understanding")
BANDS = (8192, 48640)
OUTPUT_TOKENS = 128


def render(template, item, tokenizer):
    values = {"DOC": item["context"], "Q": item["question"]}
    values.update({f"C_{letter}": item[f"choice_{letter}"] for letter in "ABCD"})
    prompt = re.sub(r"\$(DOC|Q|C_[A-D])\$", lambda match: values[match[1]].strip(), template)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def freeze(args, dataset, template, tokenizer):
    rng = random.Random(42)
    selected = []
    shortages = []
    reserves = {}
    for domain in DOMAINS:
        entries = [row for row in dataset.values() if row["domain"] == domain]
        eligible = []
        for row in entries:
            length = len(render(template, row, tokenizer))
            if BANDS[0] <= length <= BANDS[1]:
                eligible.append((length, row["_id"]))
        eligible.sort()
        middle = len(eligible) // 2
        groups = (eligible[:middle], eligible[middle:])
        print(f"{domain}: {len(eligible)} eligible under {BANDS[1]} tokens", flush=True)
        for label, group in zip(("shorter", "longer"), groups):
            if len(group) < 4:
                shortages.append({"domain": domain, "length_group": label, "available": len(group)})
            if not group:
                continue
            band = [group[0][0], group[-1][0]]
            chosen = rng.sample(group, min(4, len(group)))
            if domain == DOMAINS[0]:
                reserves[label] = sorted(set(group) - set(chosen))
            for i, (length, case_id) in enumerate(chosen):
                selected.append({"id": case_id, "domain": domain, "length_group": label,
                                 "band": band, "input_tokens": length, "screen": i == 0})
    for shortage in shortages:
        label = shortage["length_group"]
        group = reserves[label]
        for length, case_id in rng.sample(group, 4 - shortage["available"]):
            selected.append({"id": case_id, "domain": DOMAINS[0], "length_group": label,
                             "band": [group[0][0], group[-1][0]], "input_tokens": length,
                             "screen": False, "fills_shortage_in": shortage["domain"]})
    write_json(args.output, {"seed": 42, "output_tokens": OUTPUT_TOKENS, "items": selected,
                             "shortages": shortages})


def evaluate(args, dataset, template, tokenizer):
    manifest = json.loads(args.manifest.read_text())
    selected = [row for row in manifest["items"] if row["screen"] == (args.split == "screen")]
    report = {"started_at": now(), "status": "running", "results": [],
              "server_info": api(args.base_url, "/get_server_info", timeout=10)}
    write_json(args.output, report)
    started = time.monotonic()
    try:
        for case in selected:
            item = dataset[case["id"]]
            ids = render(template, item, tokenizer)
            if len(ids) != case["input_tokens"]:
                raise ValueError(f"Tokenizer parity failed for {case['id']}")
            stamp = time.monotonic()
            response = api(args.base_url, "/generate", {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": OUTPUT_TOKENS},
            }, timeout=180)
            if response["meta_info"]["prompt_tokens"] != len(ids):
                raise ValueError(f"Prompt changed during inference for {case['id']}")
            text = response["text"].replace("*", "")
            match = re.search(r"The correct answer is \(([A-D])\)", text)
            if not match:
                match = re.search(r"The correct answer is ([A-D])", text)
            answer = match[1] if match else None
            report["results"].append(dict(case, answer=answer, expected=item["answer"],
                                          correct=answer == item["answer"], response=response,
                                          capped=response["meta_info"]["finish_reason"]["type"] == "length",
                                          elapsed_s=time.monotonic() - stamp))
            write_json(args.output, report)
        report.update(status="completed", passed=sum(row["correct"] for row in report["results"]),
                      total=len(selected), capped=sum(row["capped"] for row in report["results"]))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_s=time.monotonic() - started, finished_at=now())
        write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("screen", "holdout"), default="screen")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dataset = {row["_id"]: row for row in json.loads(args.dataset.read_text())}
    template = args.template.read_text()
    if args.manifest:
        evaluate(args, dataset, template, tokenizer)
    else:
        freeze(args, dataset, template, tokenizer)


if __name__ == "__main__":
    main()
