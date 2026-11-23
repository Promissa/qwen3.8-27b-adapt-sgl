#!/usr/bin/env python3
"""Two-hour mixed-load soak against an already running server, with live telemetry."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import sys
import threading
import time
from api_smoke import cases, chat, vision_check, VisionCheckError
from common import api, gpu_sample, now, write_json
from probe import generate


def soak_evidence(result, required_cases, telemetry):
    required_cases=set(required_cases)
    counts = dict(Counter(row["name"] for row in result["requests"]))
    all_cases = set(required_cases) <= set(counts)
    vision_errors=result.get('vision_quality_errors',[])
    attempts=Counter(counts)
    attempts.update(row['name'] for row in vision_errors)
    valid_times = []
    for sample in telemetry:
        gpus = sample.get("gpus") or []
        gpu_valid = bool(gpus) and all(
            all(isinstance(gpu.get(key), (int, float)) and not isinstance(gpu.get(key), bool)
                and gpu[key] >= 0 for key in ("used_mib", "free_mib", "utilization_percent"))
            and gpu["used_mib"] + gpu["free_mib"] > 0 and gpu["utilization_percent"] <= 100
            for gpu in gpus)
        if gpu_valid and "MemAvailable:" in sample.get("host_meminfo", ""):
            valid_times.append(sample["elapsed_s"])
    elapsed = result["elapsed_s"]
    ordered = all(a <= b for a, b in zip(valid_times, valid_times[1:]))
    max_gap = max((b - a for a, b in zip(valid_times, valid_times[1:])), default=0.)
    covered = bool(valid_times) and ordered and valid_times[0] <= min(5., elapsed)
    covered = covered and elapsed - valid_times[-1] <= 5. and max_gap <= 10.
    covered = bool(covered and not result.get("monitor_errors"))
    passed = all_cases and covered and elapsed >= result["requested_seconds"] and not result.get("error") and not vision_errors
    runtime_error=bool(result.get('runtime_error') or
        (result.get('error') and not (result.get('vision_quality_failure_only') and vision_errors)))
    runtime_completed=(required_cases <= set(attempts) and covered
        and elapsed >= result['requested_seconds'] and not runtime_error)
    runtime_7200=bool(runtime_completed and result['requested_seconds']>=7200 and elapsed>=7200)
    return {"completed_case_counts": counts, "all_cases_completed": all_cases,
            'attempted_case_counts':dict(attempts), 'vision_quality_error_count':len(vision_errors),
            'runtime_stability_7200s_gate':runtime_7200,
            "telemetry_gate": {"passed": covered, "valid_samples": len(valid_times),
                "total_samples": len(telemetry), "maximum_gap_s": max_gap,
                "first_valid_s": valid_times[0] if valid_times else None,
                "last_valid_s": valid_times[-1] if valid_times else None},
            "status": (("passed" if result["vision_included"] else "passed_text_only") if passed else
                'completed_with_vision_quality_errors' if runtime_completed and vision_errors else 'failed'),
            "full_plan_soak_gate": bool(passed and result["vision_included"]
                                       and result["requested_seconds"] >= 7200 and elapsed >= 7200)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--seconds", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", help="PNG fixture for the required vision soak")
    parser.add_argument("--uncached-vision-fixtures", action="store_true",
        help="Draw fresh red-square/blue-circle fixtures with a unique frame number per vision request")
    parser.add_argument('--continue-vision-errors',action='store_true',
        help='Collect the full duration after caption mismatches; return runtime gate only, with quality/full-plan gate still failed')
    args = parser.parse_args()
    if args.seconds <= 0 or args.output.exists():
        parser.error("Require positive duration and a new output directory")
    if args.uncached_vision_fixtures and not args.image:
        parser.error("Uncached vision fixtures require a vision-enabled --image run")
    if args.continue_vision_errors and not args.image:
        parser.error('Continuing vision errors requires an --image run')
    args.output.mkdir(parents=True)
    telemetry = args.output / "telemetry.jsonl"
    stop = threading.Event()
    monitor_errors = []

    def monitor():
        while not stop.is_set():
            try:
                sample = gpu_sample()
                sample["elapsed_s"] = time.monotonic() - start
                sample["host_meminfo"] = Path("/proc/meminfo").read_text()
                with telemetry.open("a") as stream:
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
            except Exception as error:
                monitor_errors.append(f"{type(error).__name__}: {error}")
                break
            stop.wait(1)

    result = {"at": now(), "requested_seconds": args.seconds, "status": "running",
              "vision_included": bool(args.image), "requests": [],
              "vision_cache_mode": "unique_generated_frames" if args.uncached_vision_fixtures else "reused_fixture",
              "server_info": api(args.base_url, "/get_server_info"), "monitor_errors": monitor_errors}
    result['vision_quality_errors']=[]
    result['continue_vision_errors']=args.continue_vision_errors
    result["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["api_checks_sha256"] = hashlib.sha256(Path(__file__).with_name("api_smoke.py").read_bytes()).hexdigest()
    thread = threading.Thread(target=monitor, daemon=True)
    load = [("short", lambda: chat(args.base_url, "Give a one-sentence definition of a mutex.")),
            ("long", lambda: generate(args.base_url, 8192, 1024, len(result["requests"])))]
    load += cases(args.base_url, args.image)
    if args.uncached_vision_fixtures:
        # Native test geometry, redrawn from parameters rather than changing a
        # user's image. New pixels defeat decoded-image/embedding cache reuse.
        import io
        from PIL import Image, ImageDraw
        from PIL import __version__ as pillow_version
        result["pillow_version"] = pillow_version
        frames = [0]

        def fresh_vision():
            image = Image.new("RGB", (512, 320), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((45, 70, 195, 220), fill="red")
            draw.ellipse((305, 70, 455, 220), fill="blue")
            draw.text((10, 290), f"Frame {frames[0]}", fill="black")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            frames[0] += 1
            return vision_check(args.base_url, buffer.getvalue(), ("red square", "blue circle"))

        load = [(name, fresh_vision if name == "image_input" else call) for name, call in load]
    start = time.monotonic()
    thread.start()
    name, t0 = None, start
    try:
        index = 0
        while time.monotonic() - start < args.seconds:
            if monitor_errors:
                raise RuntimeError("Soak telemetry monitor failed")
            name, call = load[index % len(load)]
            t0 = time.monotonic()
            try:
                response = call()
            except VisionCheckError as error:
                if not args.continue_vision_errors or name!='image_input':
                    raise
                result['vision_quality_errors'].append({'name':name,'at':now(),
                    'elapsed_s':time.monotonic()-t0,'error':str(error),'evidence':error.evidence})
                result['elapsed_s']=time.monotonic()-start
                write_json(args.output/'soak.json',result)
                index+=1
                continue
            result["requests"].append({"name": name, "at": now(),
                "elapsed_s": time.monotonic() - t0, "response": response})
            result["elapsed_s"] = time.monotonic() - start
            write_json(args.output / "soak.json", result)
            index += 1
        result["server_info_after"] = api(args.base_url, "/get_server_info")
        if result['vision_quality_errors']:
            result['error']=f"{len(result['vision_quality_errors'])} image responses failed object checks"
            result['vision_quality_failure_only']=True
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}", full_plan_soak_gate=False)
        result['runtime_error']=f'{type(e).__name__}: {e}'
        result["failed_request"] = {"name": name, "at": now(),
                                    "elapsed_s": time.monotonic() - t0,
                                    "error": str(e)}
        if hasattr(e, "evidence"):
            result["failed_request"]["evidence"] = e.evidence
    finally:
        result["elapsed_s"] = time.monotonic() - start
        stop.set()
        thread.join(timeout=6)
        if thread.is_alive():
            monitor_errors.append("Telemetry thread did not stop")
        try:
            samples = [json.loads(line) for line in telemetry.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            monitor_errors.append(f"{type(error).__name__}: {error}")
            samples = []
        result.update(soak_evidence(result, (name for name, _ in load), samples))
        if result["status"] == "failed" and not result.get("error"):
            result["error"] = "Mixed cases, duration or GPU/host telemetry coverage incomplete"
        result["finished_at"] = now()
        write_json(args.output / "soak.json", result)
    if args.continue_vision_errors:
        return 0 if result['runtime_stability_7200s_gate'] else 1
    return 0 if result["status"].startswith("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
