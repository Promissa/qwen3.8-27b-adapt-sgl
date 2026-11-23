#!/usr/bin/env python3
"""Exercise the installed OpenCode V2 against local SGLang in a disposable fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from common import ROOT, api, fields, now, write_json

TESTS = '''import unittest
from calculator import mean
class MeanTests(unittest.TestCase):
    def test_fraction(self): self.assertAlmostEqual(mean([1, 2]), 1.5)
    def test_iterator(self): self.assertEqual(mean(iter([2, 4, 6])), 4.)
    def test_empty(self):
        with self.assertRaises(ValueError): mean([])
if __name__ == "__main__": unittest.main()
'''
BROKEN = '''def mean(values):
    """Return the arithmetic mean of a nonempty iterable."""
    return sum(values) // len(values)
'''


def audit_tool_roundtrip(path):
    """Require a completed tool result followed by another assistant response."""
    completed, errors = [], []
    final_messages = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return {"passed": False, "error": f"Malformed JSON event at line{index + 1}"}
        part = event.get("part") or {}
        if event.get("type") == "tool_use":
            row = {"index": index, "tool": part.get("tool"), "id": part.get("id"),
                   "message_id": part.get("messageID")}
            state = part.get("state") or {}
            if (part.get("type") == "tool" and row["id"] and row["tool"]
                    and row["message_id"] and state.get("status") == "completed"
                    and "output" in state):
                completed.append(row)
            elif state.get("status") == "error":
                errors.append(row)
        elif event.get("type") == "text" and part.get("text", "").strip():
            final_messages.append({"index": index, "message_id": part.get("messageID")})
    resumed = any(text["index"] > tool["index"] and text["message_id"]
                  and text["message_id"] != tool["message_id"]
                  for tool in completed for text in final_messages)
    return {"passed": bool(completed) and resumed, "completed_tools": completed,
            "failed_tools": errors, "assistant_response_after_tool": resumed,
            "scope": "Actual CLI tool result and subsequent assistant message; artifact tests checked separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new evidence directory")
    workspace = output / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "calculator.py").write_text(BROKEN)
    (workspace / "test_calculator.py").write_text(TESTS)
    config = json.loads((ROOT / "configs/opencode-v2.json").read_text())
    info = api(args.base_url, "/get_server_info")
    pools = [n for n in fields(info, "max_total_num_tokens") if isinstance(n, int) and n > 0]
    if not pools:
        raise RuntimeError("Unknown server token pool")
    config["providers"]["sglang"]["settings"]["baseURL"] = args.base_url.rstrip("/") + "/v1"
    config["providers"]["sglang"]["models"]["qwen3.8-27b"]["limit"]["context"] = min(30000, min(pools) - 512)
    (workspace / "opencode.json").write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ)
    # OpenCode run explicitly chdirs to inherited PWD, overriding Popen(cwd=).
    # Keep its location-scoped model config and tools inside this fixture.
    env["PWD"] = str(workspace)
    for kind in ("data", "config", "cache", "state"):
        env["XDG_" + kind.upper() + "_HOME"] = str(output / "opencode" / kind)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    prompt = ("Read calculator.py and test_calculator.py. Fix only calculator.py so mean accepts any "
        "finite iterable, returns the arithmetic mean, and raises ValueError for an empty iterable. "
        "Run exactly python3 -m unittest -q using the shell tool, then report the result. "
        "Work only in this fixture. Do not launch other agents or use network tools.")
    command = ["opencode", "run", "--standalone", "--model", "sglang/qwen3.8-27b",
               "--format", "json", "--title", "Local SGLang fixture", prompt]
    version = subprocess.run(["opencode", "--version"], env=env, text=True, capture_output=True, check=True).stdout.strip()
    result = {"at": now(), "status": "running", "command": command, "version": version,
              "server_info": info, "config": config, "original_source": BROKEN,
              "tests_sha256": hashlib.sha256(TESTS.encode()).hexdigest()}
    write_json(output / "harness.json", result)
    process = None
    start = time.monotonic()
    try:
        with (output / "events.jsonl").open("w") as log, (output / "stderr.log").open("w") as error:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=error, start_new_session=True)
            result["returncode"] = process.wait(timeout=args.seconds)
        validation = subprocess.run(["python3", "-m", "unittest", "-q"], cwd=workspace,
                                    text=True, capture_output=True, timeout=30)
        (output / "independent-tests.log").write_text(validation.stdout + validation.stderr)
        tests_unchanged = (workspace / "test_calculator.py").read_text() == TESTS
        changed = (workspace / "calculator.py").read_text() != BROKEN
        tool_audit = audit_tool_roundtrip(output / "events.jsonl")
        result.update(independent_test_returncode=validation.returncode, tests_unchanged=tests_unchanged,
            source_changed=changed, tool_roundtrip_audit=tool_audit,
            status="passed" if (result["returncode"] == 0 and validation.returncode == 0
                and tests_unchanged and changed and tool_audit["passed"]) else "failed")
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result["elapsed_s"] = time.monotonic() - start
        write_json(output / "harness.json", result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
