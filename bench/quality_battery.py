#!/usr/bin/env python3
"""Adapt the prior project's EN/DA/code perplexity battery to SGLang.

The full thinking GSM8K run is performed separately by eval_quality.py. This
retains the old battery's 1200-character windows, 40 windows/domain and separate
200-example greedy/nonthinking GSM8K check. SGLang input-token logprobs replace
vLLM's prompt_logprobs layout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from common import ROOT, api, now, write_json


def documents(data_root, source):
    import pyarrow.parquet as pq
    result = []
    en = pq.read_table(data_root / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    text = "".join(en)
    for start in range(0, min(len(text), 40 * 1200), 1200):
        result.append(("en", text[start:start + 1200]))
    da = pq.read_table(data_root / "fineweb2/data/dan_Latn/test/000_00000.parquet", columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(da)
    count = 0
    for text in da:
        if len(text) > 1500:
            result.append(("da", text[:1200]))
            count += 1
        if count == 40:
            break
    code = []
    for file in sorted(source.glob("*.py")):
        text = file.read_text()
        for start in range(0, min(len(text), 4 * 1200), 1200):
            if len(text[start:start + 1200]) > 800:
                code.append(("code", text[start:start + 1200]))
        if len(code) >= 40:
            break
    result.extend(code[:40])
    counts = {lang: sum(x[0] == lang for x in result) for lang in ("en", "da", "code")}
    if any(n != 40 for n in counts.values()):
        raise RuntimeError(f"Incomplete quality corpus: {counts}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--data-root", type=Path, default=ROOT / ".cache/quality-data")
    parser.add_argument("--code-source", type=Path,
        default=Path.home() / "projects/sglang/python/sglang/srt/managers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppl-only", action="store_true", help="Only the perplexity component, not the full battery")
    parser.add_argument("--windows-per-domain", type=int, default=40)
    parser.add_argument("--gsm8k-examples", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    if not 1 <= args.windows_per_domain <= 40 or not 1 <= args.gsm8k_examples <= 200:
        parser.error("Require1–40 windows/domain and1–200 greedy examples")
    corpus = documents(args.data_root, args.code_source)
    selected, seen = [], {}
    for language, text in corpus:
        if seen.get(language, 0) < args.windows_per_domain:
            selected.append((language, text))
            seen[language] = seen.get(language, 0) + 1
    corpus = selected
    digest = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest()
    result = {"at": now(), "status": "running", "corpus_sha256": digest,
              "scope": ("reduced_screening_battery" if (args.windows_per_domain, args.gsm8k_examples) != (40, 200)
                        else "perplexity_only" if args.ppl_only else "perplexity_and_200_nonthinking_gsm8k"),
              "requested_windows_per_domain": args.windows_per_domain,
              "requested_greedy_examples": 0 if args.ppl_only else args.gsm8k_examples,
              "server_info": api(args.base_url, "/get_server_info"), "windows": []}
    aggregate = {}
    try:
        for language, text in corpus:
            response = api(args.base_url, "/generate", {"text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True, "logprob_start_len": 0})
            entries = response["meta_info"].get("input_token_logprobs")
            if not entries:
                raise RuntimeError("Server omitted input-token log probabilities")
            logs = [entry[0] for entry in entries if entry[0] is not None]
            if not logs or not all(math.isfinite(x) and x <= 0 for x in logs):
                raise RuntimeError("Missing/nonfinite positive input-token log probability")
            total = sum(logs)
            stats = aggregate.setdefault(language, [0., 0])
            stats[0] += total
            stats[1] += len(logs)
            result["windows"].append({"language": language, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "logprob_sum": total, "tokens": len(logs), "meta_info": response["meta_info"]})
            write_json(args.output, result)
        result["ppl"] = {language: {"perplexity": math.exp(-total / tokens), "tokens": tokens}
                         for language, (total, tokens) in aggregate.items()}
        total_logprob = sum(x[0] for x in aggregate.values())
        total_tokens = sum(x[1] for x in aggregate.values())
        result["ppl"]["all"] = {"perplexity": math.exp(-total_logprob / total_tokens), "tokens": total_tokens}
        if not args.ppl_only:
            import pyarrow.parquet as pq
            dataset = args.data_root / "gsm8k/main/test-00000-of-00001.parquet"
            table = pq.read_table(dataset)
            questions = list(zip(table.column("question").to_pylist(), table.column("answer").to_pylist()))[:args.gsm8k_examples]
            if len(questions) != args.gsm8k_examples:
                raise RuntimeError("Incomplete greedy GSM8K battery dataset")
            result["gsm8k"] = {"n_requested": args.gsm8k_examples, "temperature": 0, "thinking": False,
                "max_tokens": 768, "corpus_sha256": hashlib.sha256(json.dumps(questions).encode()).hexdigest(),
                "examples": []}
            for question, answer in questions:
                gold = answer.split("####")[-1].strip().replace(",", "")
                response = api(args.base_url, "/v1/chat/completions", {
                    "model": "qwen3.8-27b", "messages": [{"role": "user", "content": question +
                    "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
                    "max_tokens": 768, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False}})
                text = response["choices"][0]["message"].get("content") or ""
                match = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", text)
                numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
                prediction = (match.group(1).replace(",", "") if match else
                              numbers[-1].replace(",", "") if numbers else None)
                try:
                    correct = abs(float(prediction) - float(gold)) < 1e-6
                except (TypeError, ValueError):
                    correct = False
                result["gsm8k"]["examples"].append({"question": question, "gold": gold,
                    "prediction": prediction, "correct": correct, "response": response})
                write_json(args.output, result)
            examples = result["gsm8k"]["examples"]
            result["gsm8k"]["accuracy"] = sum(x["correct"] for x in examples) / len(examples)
            result["gsm8k"]["truncated"] = sum(x["response"]["choices"][0]["finish_reason"] == "length" for x in examples)
        result["status"] = "passed"
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        write_json(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
