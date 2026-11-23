"""Freeze natural, untruncated LongBench v2 inputs near the 80% capacity point."""

import argparse
import json
from collections import Counter
from pathlib import Path

from common import ROOT, write_json
from longbench_subset import DOMAINS, evaluate, render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path)
    p.add_argument("--boundary", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--model-path", required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    cache = ROOT / ".cache/app-eval"
    template = (cache / "longbench-0shot.txt").read_text()
    items = {r["_id"]: r for r in json.loads((cache / "longbench-v2.json").read_text())}
    if args.index:
        if args.index.exists():
            p.error("Index exists")
        rows = []
        for r in items.values():
            if r["domain"] in DOMAINS:
                rows.append(
                    {
                        "id": r["_id"],
                        "domain": r["domain"],
                        "input_tokens": len(render(template, r, tok)),
                    }
                )
        write_json(args.index, rows)
        return
    if not args.boundary or not args.output or args.output.exists():
        p.error("Require a measured boundary and fresh output")
    folder = args.boundary.parent
    manifest = folder / "natural80-ids.json"
    if not manifest.exists():
        target = json.loads(args.boundary.read_text())["input80"]
        index = json.loads((folder.parent / "natural-length-index.json").read_text())
        tolerance = 0.05
        eligible = [
            r for r in index if abs(r["input_tokens"] - target) <= target * tolerance
        ]
        if not eligible:
            tolerance = 0.10
            eligible = [
                r
                for r in index
                if abs(r["input_tokens"] - target) <= target * tolerance
            ]
        if not eligible:
            raise ValueError(f"No natural cases within 10% of {target}")
        eligible.sort(key=lambda r: (abs(r["input_tokens"] - target), r["id"]))
        selected = []
        for domain in DOMAINS:
            selected.extend([r for r in eligible if r["domain"] == domain][:2])
        selected.extend([r for r in eligible if r not in selected][: 4 - len(selected)])
        write_json(
            manifest,
            {
                "target": target,
                "relative_tolerance": tolerance,
                "eligible_by_domain": dict(Counter(r["domain"] for r in eligible)),
                "items": [dict(r, screen=True) for r in selected],
            },
        )
    args.manifest = manifest
    args.split = "screen"
    args.base_url = "http://127.0.0.1:30000"
    evaluate(args, items, template, tok)


if __name__ == "__main__":
    main()
