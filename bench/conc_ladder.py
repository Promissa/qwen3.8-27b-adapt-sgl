#!/usr/bin/env python3
"""Real-prompt SGLang concurrency ladder, adapted from the prior vLLM client.

Retains every repetition. Metrics absent from SGLang remain unknown; scheduler
iterations are not model forwards, and completion counters are not live decode
counters. Client rates use actual usage and SSE timings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request
import uuid

from common import ROOT, api, now, write_json

METRICS = ("sglang:full_token_usage", "sglang:mamba_usage", "sglang:num_running_reqs",
           "sglang:num_queue_reqs", "sglang:num_retracted_requests_total",
           "sglang:spec_accept_length", "sglang:spec_accept_rate")


def parse_metrics(text):
    from prometheus_client.parser import text_string_to_metric_families
    values = {name: [] for name in METRICS}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in values:
                values[sample.name].append(sample.value)
    return {name: max(samples) if samples else None for name, samples in values.items()}


def stream(base, prompt, output_tokens, barrier):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0., "max_tokens": output_tokens, "stream": True,
        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    barrier.wait()
    start = time.perf_counter()
    first = last = usage = finish = None
    chunks = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                content = choice.get("delta", {}).get("content")
                if content:
                    last = time.perf_counter()
                    first = last if first is None else first
                    chunks.append(content)
    end = time.perf_counter()
    if not usage or first is None or last <= first or usage["completion_tokens"] < 2 or finish is None:
        raise RuntimeError("Incomplete SSE timing, usage or finish evidence")
    return {"started": start, "first_content": first, "last_content": last, "ended": end,
        "ttft_s": first - start, "elapsed_s": end - start, "usage": usage,
        "finish_reason": finish, "output": "".join(chunks),
        "decode_tokens_per_second": (usage["completion_tokens"] - 1) / (last - first)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--n", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=int, default=1024)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.n + [args.reps, args.out]) <= 0:
        parser.error("Require a new output file and positive sizes")
    corpus = (ROOT / "bench/prompts_real.jsonl").read_bytes()
    prompts = [json.loads(line)["prompt"] for line in corpus.decode().splitlines()]
    result = {"at": now(), "protocol": vars(args) | {"output": str(args.output)},
        "server_info": api(args.base_url, "/get_server_info"), "runs": [], "status": "running",
        "prompt_corpus_sha256": hashlib.sha256(corpus).hexdigest(), "prompt_count": len(prompts),
        "metric_scope": "single GPU/worker; max across any exported label series"}
    write_json(args.output, result)
    try:
        for n in args.n:
            for rep, (prompt_index, prompt) in itertools.product(range(args.reps), enumerate(prompts)):
                api(args.base_url, "/flush_cache", {})
                salt = uuid.uuid4().hex
                barrier, stop = threading.Barrier(n), threading.Event()
                telemetry, metric_errors = [], []

                def sample():
                    while not stop.is_set():
                        try:
                            with urllib.request.urlopen(args.base_url + "/metrics", timeout=5) as response:
                                raw = response.read().decode()
                            telemetry.append({"at": time.perf_counter(), "values": parse_metrics(raw), "raw": raw})
                        except Exception as e:
                            metric_errors.append(str(e))
                        stop.wait(.25)

                thread = threading.Thread(target=sample, daemon=True)
                thread.start()
                run = {"n": n, "rep": rep, "prompt_index": prompt_index,
                       "prompt": prompt, "salt": salt,
                       "streams": [], "telemetry": telemetry, "metric_errors": metric_errors}
                result["runs"].append(run)
                try:
                    with ThreadPoolExecutor(max_workers=n) as pool:
                        jobs = [pool.submit(stream, args.base_url,
                            ("" if args.shared else f"Task {salt}-{i}. ") + prompt,
                            args.out, barrier) for i in range(n)]
                        for job in jobs:
                            run["streams"].append(job.result())
                    streams = run["streams"]
                    wall = max(s["ended"] for s in streams) - min(s["started"] for s in streams)
                    overlap = min(s["last_content"] for s in streams) - max(s["first_content"] for s in streams)
                    run.update(status="passed", aggregate_e2e_tokens_per_second=sum(s["usage"]["completion_tokens"] for s in streams)/wall,
                        mean_stream_decode_tokens_per_second=statistics.mean(s["decode_tokens_per_second"] for s in streams),
                        all_streams_decode_overlap_s=max(0., overlap), all_streams_decode_overlap=overlap > 0,
                        metrics_available=bool(telemetry) and any(v is not None for row in telemetry for v in row["values"].values()))
                finally:
                    stop.set()
                    thread.join(timeout=6)
                    write_json(args.output, result)
                print(json.dumps({k: v for k, v in run.items() if k not in ("streams", "telemetry", "metric_errors")}), flush=True)
        result["status"] = "passed"
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
