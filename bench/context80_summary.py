"""Validate and summarize measurements from the 80%-context experiment."""

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path
from statistics import mean

from common import fields, write_json


def read(path):
    return json.loads(path.read_text())


def output_ids(response):
    ids = response["output_ids"]
    assert ids and len(ids) == response["meta_info"]["completion_tokens"]
    return ids


def compare_ids(left, right):
    difference = next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None
    )
    if difference is None and len(left) != len(right):
        difference = min(len(left), len(right))
    return {
        "control_tokens": len(left),
        "candidate_tokens": len(right),
        "first_difference": difference,
    }


METRICS = ("decode_tok_s", "ttft_ms", "tpot_ms", "elapsed_s", "output_throughput")


def timings(folder):
    report = read(folder / "metrics.json")
    assert report["status"] == "passed" and report["protocol"]["repeats"] == 2
    grouped = defaultdict(list)
    responses = {}
    for r in report["rows"]:
        n, rep = r["input_tokens"], r["repeat"]
        raw = folder / f"metrics-{n}-{rep}.sse.jsonl"
        with raw.open() as stream:
            for last in stream:
                pass
        response = json.loads(last)["data"]
        ids = output_ids(response)
        assert len(ids) == r["output_tokens"] == 1024
        assert (
            r["final_meta"]["prompt_tokens"] == n
            and r["final_meta"]["cached_tokens"] == 0
        )
        grouped[n].append(r)
        responses[n, rep] = ids
    result = {}
    for n, rows in grouped.items():
        assert sorted(r["repeat"] for r in rows) == [0, 1]
        metrics = {k: mean(r[k] for r in rows) for k in METRICS}
        metrics.update(
            retractions=sum(r["final_meta"]["num_retractions"] for r in rows),
            elapsed_rounds_s=[r["elapsed_s"] for r in rows],
        )
        if "spec_accept_length" in rows[0]["final_meta"]:
            metrics["accept_length"] = mean(
                r["final_meta"]["spec_accept_length"] for r in rows
            )
        result[n] = metrics
    (pool,) = fields(report["server_info"], "max_total_num_tokens")
    return pool, result, responses


def longbench(folder, names):
    rows = {}
    for name in names:
        report = read(folder / (name + ".json"))
        assert report["status"] == "completed" and report["total"] == len(
            report["results"]
        )
        for row in report["results"]:
            assert row["id"] not in rows
            assert row["response"]["meta_info"]["prompt_tokens"] == row["input_tokens"]
            rows[row["id"]] = row
    return rows


def application_stats(rows, baseline=None):
    result = dict(
        total=len(rows),
        passed=sum(r["correct"] for r in rows.values()),
        capped=sum(r["capped"] for r in rows.values()),
        request_seconds=sum(r["elapsed_s"] for r in rows.values()),
        unparsed=sum(r["answer"] is None for r in rows.values()),
        input_tokens={k: r["input_tokens"] for k, r in rows.items()},
    )
    if baseline is not None:
        assert rows.keys() == baseline.keys()
        pairs = {
            k: compare_ids(
                output_ids(baseline[k]["response"]), output_ids(r["response"])
            )
            for k, r in rows.items()
        }
        result.update(
            token_exact=sum(p["first_difference"] is None for p in pairs.values()),
            per_item_tokens=pairs,
            elapsed_speedup=sum(r["elapsed_s"] for r in baseline.values())
            / result["request_seconds"],
            same_choice=sum(
                r["answer"] == baseline[k]["answer"] for k, r in rows.items()
            ),
            gains=[
                k
                for k, r in rows.items()
                if r["correct"] and not baseline[k]["correct"]
            ],
            losses=[
                k
                for k, r in rows.items()
                if not r["correct"] and baseline[k]["correct"]
            ],
        )
    return result


def bfcl(folder):
    folder = folder / "bfcl-screen"
    report = read(folder / "summary.json")
    assert report["status"] == "completed" and report["attempted"] == 8
    generated = {
        r["id"]: r
        for path in (folder / "result").rglob("*_result.json")
        for line in path.read_text().splitlines()
        if line.strip()
        for r in [json.loads(line)]
        if "traceback" not in r
    }
    texts = {k: r["result"] for k, r in generated.items()}
    counts = [
        n for r in generated.values() for turn in r["output_token_count"] for n in turn
    ]
    assert len(texts) == report["total"]
    return dict(
        passed=report["passed"],
        scored=report["total"],
        attempted=report["attempted"],
        rejected=list(report["capacity_rejected"]),
        completed_steps=len(counts),
        steps_at_budget=sum(n == 512 for n in counts),
        inference_seconds=sum(report["inference_seconds_by_case"].values()),
        items=report["items"],
    ), texts


def retrieval(folder, expected):
    result = read(folder / "retrieval80.json")
    rows = result["checks"]
    assert result["status"] == "completed" and result["input80"] == expected
    assert sorted({r["depth"] for r in rows}) == [0.1, 0.5, 0.9]
    assert len(rows) == (6 if folder.name.startswith("gdn") else 3)
    items = {}
    for r in rows:
        for phase in ("cold", "warm"):
            response = r[phase]["response"]
            meta = response["meta_info"]
            key = (r.get("repeat", 0), r["depth"], phase)
            assert key not in items and meta["prompt_tokens"] >= expected
            if phase == "cold":
                assert meta["prompt_tokens"] == expected
            items[key] = (r["prompt_sha256"], output_ids(response))
    return dict(
        unique_keys=3,
        calls=len(items),
        correct=sum(
            r[phase]["response"]["text"].strip() == r["expected"]
            for r in rows
            for phase in ("cold", "warm")
        ),
        warm_ttft_ms=mean(r["warm"]["ttft_s"] * 1000 for r in rows),
        warm_ttft_ms_by_depth={
            d: mean(r["warm"]["ttft_s"] * 1000 for r in rows if r["depth"] == d)
            for d in (0.1, 0.5, 0.9)
        },
        cold_ttft_ms=mean(r["cold"]["ttft_s"] * 1000 for r in rows),
        warm_uncached_tokens=[
            r["warm"]["response"]["meta_info"]["prompt_tokens"]
            - r["warm"]["response"]["meta_info"]["cached_tokens"]
            for r in rows
        ],
        warm_cached_tokens=[
            r["warm"]["response"]["meta_info"]["cached_tokens"] for r in rows
        ],
    ), items


def summarize(root, family):
    reference_family = "mtp" if family == "mtp-fit" else family
    boundary = read(root / reference_family / "boundary.json")
    assert (
        boundary["status"] == "passed"
        and boundary["input80"] == boundary["max_input"] * 4 // 5
    )
    roles = [family] if family == "max" else [family + "-control", family]
    result = dict(boundary=boundary, roles={})
    observations = {}
    for role in roles:
        folder = root / role
        assert read(folder / "exit.json")["code"] == 0
        probe = read(folder / "probe/probe.json")
        assert probe["status"] == "passed_diagnostic"
        pool, timing, outputs = timings(folder)
        natural = longbench(folder, ["near80-longbench"])
        manifest = read(root / reference_family / "natural80-ids.json")
        assert set(natural) == {r["id"] for r in manifest["items"]} and natural
        record, keys = retrieval(folder, boundary["input80"])
        samples = [
            json.loads(line)["gpus"][0]["used_mib"]
            for line in (folder / "probe/gpu-telemetry.jsonl").read_text().splitlines()
            if '"gpus"' in line
        ]
        info = probe["server_info"]
        assert not info["allow_auto_truncate"]
        protocol = {
            k: info[k]
            for k in (
                "model_path",
                "mem_fraction_static",
                "attention_backend",
                "kv_cache_dtype",
                "max_mamba_cache_size",
                "chunked_prefill_size",
                "disable_radix_cache",
                "max_running_requests",
                "enable_deterministic_inference",
            )
        }
        measured = read(folder / "boundary.json")
        assert measured["pool"] == pool and measured["status"] == "passed"
        data = dict(
            settings=protocol,
            measured_max_input=measured["max_input"],
            pool=pool,
            means=timing,
            retrieval=record,
            natural80=application_stats(natural),
            peak_gpu_mib=max(samples),
            idle_gpu_mib=probe["prelaunch_gpu"]["gpus"][0]["used_mib"],
        )
        observation = dict(outputs=outputs, keys=keys, natural80=natural)
        if family in ("gdn", "mtp"):
            fixed = longbench(folder, ["longbench-screen", "longbench-holdout"])
            assert len(fixed) == 16
            b, s = bfcl(folder)
            data.update(fixed=application_stats(fixed), bfcl=b)
            data["overlap_natural_fixed"] = sorted(natural.keys() & fixed.keys())
            observation.update(fixed=fixed, bfcl=s)
        result["roles"][role] = data
        observations[role] = observation
    if len(roles) == 2:
        c, v = (result["roles"][k] for k in roles)
        assert c["pool"] == v["pool"] and c["settings"] == v["settings"]
        old, new = (observations[k] for k in roles)
        assert old["outputs"].keys() == new["outputs"].keys()
        result["speedup"] = {
            n: c["means"][n]["elapsed_s"] / v["means"][n]["elapsed_s"]
            for n in c["means"]
        }
        result["timed_token_comparison"] = {
            f"{n}-{r}": compare_ids(old["outputs"][n, r], ids)
            for (n, r), ids in new["outputs"].items()
        }
        assert old["keys"].keys() == new["keys"].keys()
        assert all(old["keys"][k][0] == val[0] for k, val in new["keys"].items())
        v["retrieval"]["token_exact"] = sum(
            old["keys"][k][1] == val[1] for k, val in new["keys"].items()
        )
        for group in ("fixed", "natural80") if "fixed" in old else ("natural80",):
            v[group] = application_stats(new[group], old[group])
        if "bfcl" not in old:
            return result
        common = old["bfcl"].keys() & new["bfcl"].keys()
        v["bfcl"].update(
            common_scored=len(common),
            identical_turn_outputs=sum(
                old["bfcl"][k] == new["bfcl"][k] for k in common
            ),
            gains=sorted(
                k for k in common if v["bfcl"]["items"][k] and not c["bfcl"]["items"][k]
            ),
            losses=sorted(
                k for k in common if not v["bfcl"]["items"][k] and c["bfcl"]["items"][k]
            ),
        )
    return result


def experiment(root, family):
    # A failed uncapped allocation may be followed by bounded pool sizes.
    locations = [
        root,
        *sorted(
            root.glob(f"fraction-{family}-*"),
            key=lambda x: int(x.name.rsplit("-", 1)[1]),
            reverse=True,
        ),
    ]
    attempts = []
    chosen = None
    for folder in locations:
        candidate = folder / family
        if not (candidate / "exit.json").exists():
            continue
        bfile = candidate / "boundary.json"
        b = read(bfile) if bfile.exists() else {}
        code = read(candidate / "exit.json")["code"]
        item = dict(
            path=str(candidate.relative_to(root)),
            returncode=code,
            pool=b.get("pool"),
            threshold=b.get("scheduler_threshold"),
            boundary_status=b.get("status"),
            trials=b.get("trials", []),
        )
        if code:
            log = (candidate / "probe/server.log").read_text()
            pending = re.findall(r"#pending-token: (\d+)", log)
            if pending:
                item["last_logged_pending_tokens"] = int(pending[-1])
            validation = candidate / "probe/validation-00.log"
            if validation.exists():
                item["validation_log_tail"] = validation.read_text()[-500:]
        attempts.append(item)
        if code == 0 and chosen is None:
            chosen = folder
    if chosen is None:
        raise RuntimeError(f"No completed configuration for {family}")
    result = summarize(chosen, family)
    result["attempts"] = attempts
    result["selected_directory"] = str((chosen / family).relative_to(root))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--families", nargs="+", default=["mtp", "gdn", "max", "mtp-fit"])
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    write_json(
        args.output or args.root / "comparison.json",
        {name: experiment(args.root, name) for name in args.families},
    )


if __name__ == "__main__":
    main()
