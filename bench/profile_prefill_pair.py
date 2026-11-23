"""Profile one cold 8K input and one 40-token extend with a 48K shared prefix."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random

from common import ROOT, api, write_json
from long_context import make_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        ROOT / "weights/target-embedding-nvfp4", local_files_only=True
    )
    prefix = make_prompt(tok, 48640, 0.5, "ARCHIVE_48640_1_74309261")
    base = "http://127.0.0.1:30000"

    def generate(ids):
        return api(
            base,
            "/generate",
            dict(input_ids=ids, sampling_params=dict(temperature=0, max_new_tokens=1)),
        )

    rng = random.Random(9893)
    for tag, ids in [
        ("cold", [rng.randrange(1000, 20000) for _ in range(8192)]),
        ("warm", prefix + list(range(2000, 2040))),
    ]:
        api(base, "/flush_cache", {})
        if tag == "warm":
            generate(prefix)
        out = (args.output / tag).resolve()
        out.mkdir()
        api(
            base,
            "/start_profile",
            {
                "output_dir": str(out),
                "activities": ["CPU", "GPU"],
                "profile_id": tag,
                "with_stack": False,
                "record_shapes": False,
            },
        )
        try:
            response = generate(ids)
        finally:
            api(base, "/stop_profile", {}, timeout=180)
        write_json(out / "response.json", response)
        totals = Counter()
        for path in out.glob("*.trace.json.gz"):
            with gzip.open(path, "rt") as f:
                events = json.load(f)["traceEvents"]
            for e in events:
                if e.get("cat") == "kernel":
                    totals[e["name"]] += e["dur"]
        if not totals:
            raise RuntimeError(f"Missing GPU events for {tag}")
        write_json(
            out / "kernels.json",
            {
                "summed_ms": sum(totals.values()) / 1000,
                "kernel_ms": {k: v / 1000 for k, v in totals.most_common()},
                "cached": response["meta_info"]["cached_tokens"],
            },
        )


if __name__ == "__main__":
    main()
