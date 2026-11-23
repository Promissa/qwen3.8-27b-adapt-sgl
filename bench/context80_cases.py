"""Repeat the original 1K/8K protocol at the measured boundary's 80% point."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from common import ROOT, api, fields, write_json
from long_context import make_prompt, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--boundary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--focus-80", action="store_true")
    args = p.parse_args()
    boundary = json.loads(args.boundary.read_text())
    assert boundary["status"] == "passed"
    if args.output.name.endswith("-control") or args.focus_80:
        # Touch the same high-water KV pages before timing both sides of the pair.
        subprocess.run(
            [
                sys.executable,
                "bench/context_limit.py",
                "--output",
                str(args.output / "boundary.json"),
            ],
            check=True,
            cwd=ROOT,
        )
    length = boundary["input80"]
    base = "http://127.0.0.1:30000"
    info = api(base, "/get_server_info")
    (pool,) = fields(info, "max_total_num_tokens")
    assert length + 1024 < pool and not info["allow_auto_truncate"]
    lengths = [length] if args.focus_80 else [1024, 8192, length]
    common = args.boundary.parent.parent / "mtp/boundary.json"
    if args.boundary.parent.name == "gdn" and common.exists():
        other_length = json.loads(common.read_text())["input80"]
        if other_length not in lengths and other_length + 1024 < pool:
            lengths.append(other_length)
    subprocess.run(
        [
            sys.executable,
            "bench/target_metrics.py",
            "--lengths",
            ",".join(map(str, lengths)),
            "--repeats",
            "2",
            "--output",
            str(args.output / "metrics.json"),
        ],
        check=True,
        cwd=ROOT,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    (newline,) = tok.encode("\n", add_special_tokens=False)
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
    report = dict(status="running", input80=length, checks=[])
    order = [0, 1, 2, 2, 1, 0] if args.boundary.parent.name == "gdn" else [0, 1, 2]
    for sample, i in enumerate(order):
        depth = (0.1, 0.5, 0.9)[i]
        key = f"ARCHIVE_{length}_{i}_74309261"
        ids = make_prompt(tok, length, depth, key)
        # Fill the <one filler-line remainder within the document, before its question.
        pad = length - len(ids)
        assert 0 <= pad <= 64
        ids[100:100] = [newline] * pad
        assert len(ids) == length
        api(base, "/flush_cache", {})
        cold = stream(base, ids)
        assert cold["response"]["meta_info"]["prompt_tokens"] == length
        gen = list(cold["response"]["output_ids"])
        if gen and gen[-1] == tok.eos_token_id:
            gen.pop()
        warm = stream(base, ids + gen + suffix)
        report["checks"].append(
            dict(
                depth=depth,
                repeat=sample // 3,
                expected=key,
                prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                cold=cold,
                warm=warm,
                correct=all(v["response"]["text"].strip() == key for v in (cold, warm)),
            )
        )
        write_json(args.output / "retrieval80.json", report)
    report["status"] = "completed"
    write_json(args.output / "retrieval80.json", report)
    subprocess.run(
        [
            sys.executable,
            "bench/longbench_near_limit.py",
            "--boundary",
            str(args.boundary),
            "--model-path",
            args.model_path,
            "--output",
            str(args.output / "near80-longbench.json"),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
