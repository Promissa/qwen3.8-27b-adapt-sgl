#!/usr/bin/env python3
"""Run frozen multi-turn tasks using BFCL's generator and official checker."""
import argparse
import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from common import now, write_json

CATEGORIES = ("multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param")
MODEL = "qwen3.8-27b"
TOKEN_CAP = 512


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def call_simulator(expression):
    """Execute a BFCL simulator method with literal arguments."""
    caller = sys._getframe(1)
    call = ast.parse(expression, mode="eval").body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        raise ValueError("Expected a simulator method call")
    target = call.func
    allowed = caller.f_locals["class_method_name_mapping"]
    if not isinstance(target.value, ast.Name) or allowed.get(target.attr) != target.value.id:
        raise ValueError("Unknown simulator method")
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    if None in keywords or len(keywords) != len(call.keywords):
        raise ValueError("Expected unique literal keyword arguments")
    simulator = caller.f_globals[target.value.id]
    return getattr(simulator, target.attr)(*(ast.literal_eval(arg) for arg in call.args), **keywords)


def backend():
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval.__main__ import cli

    class LocalHandler(OpenAICompletionsHandler):
        def generate_with_backoff(self, **kwargs):
            kwargs.update(max_completion_tokens=TOKEN_CAP,
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return super().generate_with_backoff(**kwargs)

    MODEL_CONFIG_MAPPING[MODEL] = ModelConfig(
        model_name=MODEL, display_name=MODEL, url="", org="Qwen", license="Apache-2.0",
        model_handler=LocalHandler, is_fc_model=True, underscore_to_dot=True,
    )
    multi_turn_utils.eval = call_simulator
    sys.argv.pop(1)
    cli()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("screen", "holdout", "all"), default="screen")
    parser.add_argument("--score-retained", type=Path, help="Score saved BFCL output without a model request")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    if not args.manifest:
        if not args.data_dir:
            parser.error("--data-dir is required for freezing")
        rng = random.Random(42)
        ids = {}
        for category in CATEGORIES:
            entries = jsonl(args.data_dir / f"BFCL_v4_{category}.json")
            ids[category] = rng.sample(sorted(row["id"] for row in entries), 8)
        write_json(args.output, {"seed": 42, "ids": ids, "output_tokens_per_step": TOKEN_CAP})
        return

    output = args.output.resolve()
    output.mkdir(parents=True)
    ids = json.loads(args.manifest.read_text())["ids"]
    subset = {"screen": slice(2), "holdout": slice(2, None), "all": slice(None)}[args.split]
    selections = {category: values[subset] for category, values in ids.items() if values[subset]}
    if not selections:
        parser.error("No BFCL IDs in the selected split")
    write_json(output / "test_case_ids_to_generate.json", selections)
    env = dict(os.environ, BFCL_PROJECT_ROOT=str(output), OPENAI_API_KEY="local",
               OPENAI_BASE_URL="http://127.0.0.1:30000/v1")
    for key in ("OPENAI_DEFAULT_HEADERS", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                "HTTPS_PROXY", "https_proxy"):
        env.pop(key, None)
    runner = [sys.executable, str(Path(__file__).resolve()), "_backend"]
    started = time.monotonic()
    report = {"started_at": now(), "status": "running"}
    write_json(output / "summary.json", report)
    try:
        if args.score_retained:
            report["retained_run"] = str(args.score_retained.resolve())
            raw_dir = args.score_retained / "result"
        else:
            subprocess.run([*runner, "generate", "--model", MODEL, "--run-ids", "--temperature", "0",
                            "--num-threads", "1"], env=env, check=True)
            raw_dir = output / "result"
        case_times = {}
        rejected = {}
        scored_dir = output / "scored-input"
        for category, selected in selections.items():
            result_file, = raw_dir.rglob(f"BFCL_v4_{category}_result.json")
            generated = jsonl(result_file)
            if sorted(row["id"] for row in generated) != sorted(selected):
                raise ValueError(f"Incomplete generation for {category}")
            for row in generated:
                if "traceback" in row:
                    message = row["result"]
                    if "Input length" not in message or "exceeds the maximum allowed length" not in message:
                        raise ValueError(f"Inference failure for {row['id']}: {message}")
                    rejected[row["id"]] = message
                    continue
                case_times[row["id"]] = sum(sum(turn) for turn in row["latency"])
            scored = [row for row in generated if row["id"] in case_times]
            if scored:
                dest = scored_dir / result_file.relative_to(raw_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text("".join(json.dumps(row) + "\n" for row in scored))
        subprocess.run([*runner, "evaluate", "--model", MODEL, "--test-category",
                        ",".join(selections), "--result-dir", str(scored_dir), "--partial-eval"],
                       env=env, check=True)
        items = dict.fromkeys(rejected)
        for category, selected in selections.items():
            selected = [case for case in selected if case not in rejected]
            if not selected:
                continue
            score_file, = (output / "score").rglob(f"BFCL_v4_{category}_score.json")
            header, *failures = jsonl(score_file)
            invalid = {row["id"] for row in failures}
            if (not invalid <= set(selected) or header["total_count"] != len(selected)
                    or header["correct_count"] != len(selected) - len(invalid)):
                raise ValueError(f"Unexpected scorer coverage for {category}")
            items.update({case: case not in invalid for case in selected})
        report.update(status="completed", passed=sum(value is True for value in items.values()),
                      total=len(case_times), attempted=len(items), items=items, capacity_rejected=rejected,
                      inference_seconds_by_case=case_times)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_s=time.monotonic() - started, finished_at=now())
        write_json(output / "summary.json", report)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_backend"]:
        backend()
    else:
        main()
