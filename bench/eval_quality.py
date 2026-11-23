#!/usr/bin/env python3
"""Use the upstream sgl-eval bridge, full GSM8K and GPQA (no custom scorer)."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from common import ROOT, api, now, write_json

DEFAULT_MAX_TOKENS = 16384


def gsm8k_preflight(base_url, max_tokens, effort, output, num_examples=1319):
    """Reject an undersized pool before spending a full matched evaluation."""
    import hashlib
    import json
    import sgl_eval.evals._loader as loader
    from sgl_eval.evals._math import render_math_prompt
    from sgl_eval.evals._prompts import resolve_prompt
    from sgl_eval.registry import get
    from transformers import AutoTokenizer
    from probe import capacity

    if output.exists():
        raise FileExistsError(f"Use a new quality tag; preflight already exists: {output}")
    loader._CACHE_ROOT = ROOT / ".cache/sgl_eval"
    examples = loader.load_via_prepare("gsm8k", ("test",))(num_examples)
    if len(examples) != num_examples:
        raise RuntimeError("Incomplete requested GSM8K preflight dataset")
    info = api(base_url, "/get_server_info")
    tokenizer_path = info.get("tokenizer_path") or info["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    template = resolve_prompt("math")
    system = get("gsm8k").default_gen.system_message
    chat_kwargs = {"enable_thinking": True}
    if effort:
        chat_kwargs["reasoning_effort"] = effort
    lengths = []
    for example in examples:
        messages = [{"role": "user", "content": render_math_prompt(template, example.inputs["problem"])}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        lengths.append(len(tokenizer.apply_chat_template(messages, tokenize=True,
            return_dict=False, add_generation_prompt=True, **chat_kwargs)))
    available = capacity(info, {})
    required = max(lengths) + max_tokens
    passed = available >= required
    write_json(output, {"at": now(), "status": "passed" if passed else "failed",
        "n": len(examples), "tokenizer_path": tokenizer_path, "chat_template_kwargs": chat_kwargs,
        "max_output_tokens": max_tokens, "max_prompt_tokens": max(lengths),
        "required_capacity": required, "reported_capacity": available,
        "prompt_lengths": lengths, "prompt_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(json.dumps([e.inputs for e in examples], sort_keys=True).encode()).hexdigest(),
        "full_dataset": num_examples == 1319,
        "scope": f"Requested{num_examples} actual upstream prompts and current local tokenizer; no quality score"})
    if not passed:
        raise RuntimeError(f"Full GSM8K needs {required} tokens; measured capacity {available}. No cap reduction or silent clamping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--eval", choices=["gsm8k", "gpqa"], default="gsm8k")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Fixed output cap; keep identical across compared recipes")
    parser.add_argument("--num-examples", type=int,
                        help="Explicit reduced screening set; default remains the full dataset")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "xhigh"],
                        help="Native Qwen3.8 template steering; separate from weight comparisons")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    full_count = 1319 if args.eval == "gsm8k" else 198
    count = full_count if args.num_examples is None else args.num_examples
    if not 1 <= count <= full_count:
        parser.error(f"--num-examples must be between1 and{full_count}")
    options = dict(eval_name=args.eval, model="qwen3.8-27b", base_url=args.base_url,
        num_examples=count,
        num_threads=1, max_tokens=args.max_tokens, temperature=1.0,
        top_p=.95, top_k=20, repeat=1, seed=1701, thinking_mode="qwen-3",
        sgl_eval_out_dir=str(ROOT / "results" / ("gsm8k" if args.eval == "gsm8k" else "quality") / args.tag))
    if args.reasoning_effort:
        # Keep the effective steering in upstream's recorded generation options.
        options["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    if args.dry_run:
        import json
        print(json.dumps(options, indent=2))
        return
    if args.eval == "gsm8k":
        gsm8k_preflight(args.base_url, args.max_tokens, args.reasoning_effort,
                       ROOT / "results/gsm8k" / args.tag / "preflight.json", count)
    # sgl-eval 0.1.3 has no cache-directory option; keep its prepared datasets
    # inside this experiment workspace without changing the scorer or data.
    import sgl_eval.evals._loader as loader
    loader._CACHE_ROOT = ROOT / ".cache/sgl_eval"
    from sglang.test.sgl_eval_utils import run_sgl_eval
    run_sgl_eval(SimpleNamespace(**options))


if __name__ == "__main__":
    main()
