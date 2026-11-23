#!/usr/bin/env python3
"""Own one server process group, record evidence, and test the full memory gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from common import ROOT, api, fields, gpu_sample, now, write_json

PROJECT_ENV = ROOT / '.venv'
PROJECT_PYTHON = PROJECT_ENV / 'bin/python'


def validation_command(command):
    """Use the project interpreter for owned Python validation scripts."""
    command=list(command)
    if len(command)<2 or Path(command[0]).name not in ('python','python3','python3.12') or not PROJECT_PYTHON.is_file():
        return command
    script=Path(command[1])
    if not script.is_absolute():script=ROOT/script
    try:
        script.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return command
    if script.suffix=='.py' and script.is_file():
        command[0]=str(PROJECT_PYTHON)
    return command


def runtime_provenance(output):
    """Snapshot the imported engine, independently of historical setup notes."""
    result = {"at": now(), "python": sys.version, "executable": sys.executable,
              "packages": {}}
    for name in ("sglang", "torch", "flashinfer-python", "transformers", "sgl-eval"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    spec = importlib.util.find_spec("sglang")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    result["sglang_import_origin"] = str(origin) if origin else None
    if origin:
        source = origin.parents[2]
        if (source / "python/sglang/__init__.py").resolve() == origin:
            revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=False)
            if revision.returncode == 0:
                diff = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
                patch = Path(output) / "engine.patch"
                patch.write_bytes(diff)
                result.update(engine_source=str(source), engine_commit=revision.stdout.strip(),
                              engine_diff_sha256=hashlib.sha256(diff).hexdigest(),
                              engine_patch=str(patch))
    result["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def parse_log(text):
    evidence = {}
    for key in ("max_total_num_tokens", "max_running_requests", "max_mamba_cache_size"):
        values = re.findall(r"\b" + key + r"[=: ]+(\d+)", text)
        evidence[key] = int(values[-1]) if values else None
    evidence["memory_lines"] = [line for line in text.splitlines() if re.search(
        r"(avail.*mem|available.*mem|memory pool|state.*pool|mamba cache|reserv|Post-capture|post-capture|KV cache)",
        line, re.I)]
    return evidence


def generate(base, length, output, salt=0):
    # Synthetic IDs measure capacity only; they are never quality/acceptance evidence.
    rng = random.Random(1701 + salt)
    inputs = [rng.randrange(1000, 20000) for _ in range(length)]
    start = time.perf_counter()
    response = api(base, "/generate", {"input_ids": inputs,
        "sampling_params": {"temperature": 0, "max_new_tokens": output,
                            "min_new_tokens": output, "ignore_eos": True}})
    meta = response["meta_info"]
    if meta.get("prompt_tokens") != length or meta.get("completion_tokens") != output:
        raise RuntimeError(f"Truncated capacity test: expected {length}/{output}, got {meta}")
    return {"input_tokens": length, "output_tokens": output,
            "elapsed_s": time.perf_counter() - start, "meta_info": meta}


def concurrency_activity(telemetry, requested):
    active = [row["at"] for row in telemetry
              if row["running_requests"] is not None
              and row["running_requests"] >= requested]
    observed = [row["running_requests"] for row in telemetry
                if row["running_requests"] is not None]
    span = max(active) - min(active) if active else 0.
    return {"requested": requested, "peak_running_requests": max(observed) if observed else None,
            "samples_at_requested_concurrency": len(active), "activity_span_s": span,
            "passed": len(active) >= 2 and span >= .5}


def concurrent_capacity(base, requested):
    from conc_ladder import parse_metrics

    barrier, stop = threading.Barrier(requested), threading.Event()
    telemetry, errors = [], []

    def sample():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(base + "/metrics", timeout=2) as response:
                    raw = response.read().decode()
                telemetry.append({"at": time.perf_counter(), "running_requests":
                                  parse_metrics(raw)["sglang:num_running_reqs"], "raw": raw})
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
            stop.wait(.2)

    def request(salt):
        barrier.wait()
        return generate(base, 8192, 1024, salt)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        with ThreadPoolExecutor(requested) as pool:
            outputs = list(pool.map(request, range(requested)))
    finally:
        stop.set()
        thread.join(timeout=3)
    activity = concurrency_activity(telemetry, requested)
    activity.update(telemetry=telemetry, metric_errors=errors,
                    scope="Single GPU/worker active requests, not client queue concurrency")
    return outputs, activity


def capacity(info, log):
    pools = [x for x in fields(info, "max_total_num_tokens") if isinstance(x, int) and x > 0]
    contexts = [x for x in fields(info, "context_length") if isinstance(x, int) and x > 0]
    if not pools and log.get("max_total_num_tokens"):
        pools = [log["max_total_num_tokens"]]
    if not pools:
        raise RuntimeError("Server did not report token pool capacity; near-limit gate cannot be verified")
    return min(pools + contexts)


def run(args, launch):
    sys.path.insert(0, str(ROOT / "scripts"))
    from serve import effective_argument, validate_checkpoints
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "probe.json"
    if result_path.exists():
        raise RuntimeError(f"Refusing to overwrite evidence: {result_path}")
    result = {"started_at": now(), "status": "running", "command": launch,
              "launcher_executable": os.environ.get('QWEN_PROBE_LAUNCHER_PYTHON',sys.executable),
              "capacity_gate_passed": False,
              "desktop": args.desktop, "concurrency": args.concurrency,
              "gpu_before": gpu_sample(), "environment": {}, "requests": []}
    result["runtime_provenance"] = runtime_provenance(args.output)
    result["environment_scope"] = "Historical setup snapshot; runtime_provenance identifies this launch"
    for path in (ROOT / "results/environment.json",):
        if path.exists():
            result["environment"] = json.loads(path.read_text())
    result["experiment_environment"] = {k: v for k, v in os.environ.items()
                                        if k.startswith("SGLANG_") or k == "MAX_JOBS"}
    for name, flag in (("target", "--model-path"), ("draft", "--speculative-draft-model-path")):
        value = effective_argument(launch, flag)
        if value is None:
            continue
        path = Path(value)
        if (path / "revision.json").exists():
            result[name + "_revision"] = json.loads((path / "revision.json").read_text())
        if (path / "conversion.json").exists():
            result[name + "_conversion"] = json.loads((path / "conversion.json").read_text())
    write_json(result_path, result)
    samples = []
    stop = threading.Event()
    process = None
    base = f"http://127.0.0.1:{args.port}"
    log_path = args.output / "server.log"

    def monitor():
        while not stop.is_set():
            samples.append(gpu_sample())
            stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    try:
        validate_checkpoints(launch)
        # Do not mistake somebody else's server for this probe's child.
        import socket
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # A connect to an unused loopback port can wait for TCP retries in
            # WSL's mirrored networking mode. Binding checks ownership directly
            # and does not send a potentially black-holed SYN to Windows.
            try:
                sock.bind(("127.0.0.1", args.port))
            except OSError as e:
                raise RuntimeError(f"Port {args.port} cannot be bound: {e}") from e
        thread.start()
        with log_path.open("w") as log:
            process = subprocess.Popen(launch, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + args.ready_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited before READY with code {process.returncode}")
                try:
                    # /health may return plain text rather than JSON.
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Server READY timeout")
            result["ready_s"] = time.monotonic() - (deadline - args.ready_timeout)
            result["server_info"] = api(base, "/get_server_info", timeout=30)
            log.flush()
            result["startup_log"] = parse_log(log_path.read_text(errors="replace"))
            limit = capacity(result["server_info"], result["startup_log"])
            result["reported_context_capacity"] = limit
            if limit < 8192 + 1024:
                raise RuntimeError(f"Reported capacity {limit} cannot fit 8192/1024")
            api(base, "/flush_cache", {})
            result["requests"].append(generate(base, max(8192, int(limit * .95) - 1), 1))
            api(base, "/flush_cache", {})
            if args.concurrency == 1:
                result["requests"].append(generate(base, 8192, 1024))
            else:
                outputs, activity = concurrent_capacity(base, args.concurrency)
                result["requests"].extend(outputs)
                result["concurrency_activity"] = activity
                if activity["peak_running_requests"] is None:
                    raise RuntimeError("Concurrency telemetry missing; capacity is unverified")
                if not activity["passed"]:
                    raise RuntimeError(f"Concurrent capacity gate did not observe {args.concurrency} active requests for 0.5s")
            result["server_info_after"] = api(base, "/get_server_info", timeout=30)
            result["capacity_gate_passed"] = True
            write_json(result_path, result)
            if getattr(args, "validation_plan", None):
                steps = json.loads(args.validation_plan.read_text())
                result["validation_steps"] = []
                for index, step in enumerate(steps):
                    requested_cmd = step["command"]
                    cmd = validation_command(requested_cmd)
                    with (args.output / f"validation-{index:02d}.log").open("w") as step_log:
                        completed = subprocess.run(cmd, cwd=ROOT, stdout=step_log,
                            stderr=subprocess.STDOUT, timeout=step.get("timeout_s", 86400), check=False)
                    evidence={"command":cmd,"returncode":completed.returncode}
                    if cmd!=requested_cmd:evidence['requested_command']=requested_cmd
                    result["validation_steps"].append(evidence)
                    write_json(result_path, result)
                    if completed.returncode:
                        raise RuntimeError(f"Validation step {index} failed with {completed.returncode}")
            if process.poll() is not None:
                raise RuntimeError("Server died after requests")
            result["status"] = "passed"
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            result["server_returncode"] = process.returncode
        stop.set()
        if thread.ident:
            thread.join(timeout=6)
        write_json(args.output / "gpu.json", samples)
        peaks = [g["used_mib"] for s in samples for g in s.get("gpus", [])]
        result["peak_device_used_mib"] = max(peaks) if peaks else None
        result["peak_measurement"] = "whole-device nvidia-smi sampled at 1s, not allocator peak"
        result["finished_at"] = now()
        if log_path.exists():
            result["log_evidence"] = parse_log(log_path.read_text(errors="replace"))
        write_json(result_path, result)
    print(json.dumps({"status": result["status"], "result": str(result_path),
                      "error": result.get("error")}))
    return 0 if result["status"] == "passed" else 1


def main():
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("MAX_JOBS", "2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--desktop", choices=["present", "absent", "unknown"], default="unknown")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-plan", type=Path, help="JSON list of command/timeout_s steps run before shutdown")
    args, extra = parser.parse_known_args()
    if args.concurrency < 1 or args.ready_timeout <= 0:
        parser.error("Concurrency and READY timeout must be positive")
    if extra[:1] == ["--"]:
        extra = extra[1:]
    sys.path.insert(0, str(ROOT / "scripts"))
    from serve import command
    launch = command(args.profile, extra + ["--port", str(args.port),
        "--max-running-requests", str(args.concurrency),
        "--cuda-graph-max-bs-decode", str(args.concurrency)])
    if args.concurrency > 1 and "--enable-metrics" not in launch:
        launch += ["--enable-metrics"]
    if args.profile == "dspark":
        launch += ["--max-mamba-cache-size", str(4 * args.concurrency + 4)]
    if args.dry_run:
        print(json.dumps(launch))
        return 0
    return run(args, launch)


if __name__ == "__main__":
    # Already-running controllers may still pass their older interpreter in
    # argv. Re-exec before constructing commands or recording provenance.
    if PROJECT_PYTHON.is_file() and Path(sys.prefix).resolve()!=PROJECT_ENV.resolve():
        os.environ['QWEN_PROBE_LAUNCHER_PYTHON']=sys.executable
        os.execv(str(PROJECT_PYTHON),[str(PROJECT_PYTHON),str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.exit(main())
