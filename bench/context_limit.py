"""Measure the serving input boundary with auto-truncation off and one output token."""

import argparse
import random
import time
import urllib.error
from pathlib import Path

from common import api, fields, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output")
    base = "http://127.0.0.1:30000"
    info = api(base, "/get_server_info")
    (pool,) = fields(info, "max_total_num_tokens")
    (threshold,) = fields(info, "max_req_input_len")
    assert not info["allow_auto_truncate"]
    report = dict(
        status="running",
        pool=pool,
        scheduler_threshold=threshold,
        mem_fraction_static=info["mem_fraction_static"],
        trials=[],
    )
    write_json(args.output, report)
    rng = random.Random(1189)
    ids = [rng.randrange(1000, 20000) for _ in range(threshold)]
    # Probe the excluded boundary first. The pinned scheduler uses >= for rejection.
    for n in (threshold, threshold - 1):
        api(base, "/flush_cache", {})
        start = time.monotonic()
        try:
            d = api(
                base,
                "/generate",
                {
                    "input_ids": ids[:n],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                },
                timeout=180,
            )
            m = d["meta_info"]
            row = dict(
                length=n,
                actual=m["prompt_tokens"],
                output=m["completion_tokens"],
                finish=m["finish_reason"],
                ids=d["output_ids"],
            )
            row["ok"] = (
                row["actual"] == n
                and row["output"] == 1
                and m["finish_reason"]["type"] != "abort"
            )
        except urllib.error.HTTPError as error:
            row = dict(
                length=n,
                ok=False,
                http=error.code,
                message=error.read().decode()[:1600],
            )
        except (TimeoutError, OSError) as error:
            row = dict(length=n, ok=False, error=f"{type(error).__name__}: {error}")
        row["elapsed_s"] = time.monotonic() - start
        report["trials"].append(row)
        write_json(args.output, report)
    rejected, accepted = report["trials"]
    if not accepted["ok"]:
        report["status"] = "failed"
        write_json(args.output, report)
    assert (
        not rejected["ok"]
        and rejected.get("http") == 400
        and "Input length" in rejected.get("message", "")
        and accepted["ok"]
    ), report
    report.update(
        status="passed", max_input=threshold - 1, input80=(threshold - 1) * 4 // 5
    )
    write_json(args.output, report)


if __name__ == "__main__":
    main()
