"""Run a reviewed public engineering issue with local Claude Code and export its patch."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib
import uuid

CACHE = Path(".cache/agent-repair").resolve()
PRO = {
    "duration": ("swe-case.json", "qutebrowser-96b99780"),
    "fonts": ("swe-fonts-case.jsonl", "qutebrowser-ff1c025"),
    "suggest": ("swe-suggest-case.jsonl", "qutebrowser-a84ecf"),
}
DEEP = {
    "slices": "abs-stepped-slices",
    "sqlfmt": "sqlfmt-create-table-ddl-formatting",
    "tomlkit": "tomlkit-toml-table-converters",
}
GO_MODULE_CACHE = (
    "install -d -o evalrun -g evalrun /home/evalrun/go/pkg && "
    "cp -a /root/go/pkg/mod /home/evalrun/go/pkg/ && "
    "chown -R evalrun:evalrun /home/evalrun/go"
)
MODEL = "qwen3.8-27b"

# Task images have Python; some omit curl. Check both hops before starting the agent.
CHECK_RELAY = """
import http.client, time
for attempt in range(20):
    try:
        connection = http.client.HTTPConnection('127.0.0.1', 30020, timeout=2)
        connection.request('GET', '/')
        assert connection.getresponse().status == 404
        break
    except OSError:
        time.sleep(0.5)
else:
    raise RuntimeError('Local model adapter unreachable')
"""


def task_config(case):
    if case in PRO:
        path, expected = PRO[case]
        row = json.loads((CACHE / path).read_text())
        if expected not in row["instance_id"]:
            raise ValueError(f"Unexpected instance in {path}")
        prompt = "\n\n".join(row[k] for k in ("problem_statement", "requirements", "interface"))
        return (row["instance_id"], row["base_commit"], "jefzda/sweap-images:" + row["dockerhub_tag"],
                prompt, 10800)
    task = CACHE / "deep/tasks" / DEEP[case]
    info = tomllib.loads((task / "task.toml").read_text())
    return (info["task"]["name"], info["metadata"]["base_commit_hash"],
            info["environment"]["docker_image"], (task / "instruction.md").read_text(),
            info["agent"]["timeout_sec"])


def run_case(case, output):
    instance, base, image, prompt, deadline = task_config(case)
    output.mkdir(exist_ok=False, parents=True)
    name = f"qwen-eval-{case}-{uuid.uuid4().hex[:8]}"
    cli = Path(shutil.which("claude") or "/home/pr/.local/bin/claude").resolve(strict=True)
    version = subprocess.check_output([str(cli), "--version"], text=True).strip()
    subprocess.run([
        "docker", "run", "-d", "--name", name, "--network", "none", "--cpus", "2",
        "--memory", "8g", "-v", f"{cli}:/usr/local/bin/claude:ro",
        "-v", f"{CACHE}/model-socket:/model-connect:ro",
        "-v", f"{CACHE}/container-adapter.py:/tmp/eval-listener.py:ro",
        "--entrypoint", "sleep", image, "infinity",
    ], stdout=subprocess.DEVNULL, check=True)
    start = time.monotonic()
    try:
        container = ["docker", "exec", name]
        git = [*container, "runuser", "-u", "evalrun", "--", "git", "-C", "/app"]
        subprocess.run([*container, "sh", "-ec", "useradd -m evalrun && chown -R evalrun /app"], check=True)
        subprocess.run(["docker", "exec", "-d", name, "python3", "/tmp/eval-listener.py", "task",
                        "/model-connect/messages.sock"], check=True)
        if case == "slices":
            # The stock offline Go image places its module cache under /root.
            subprocess.run([*container, "sh", "-ec", GO_MODULE_CACHE], check=True)
        checkout = ["-B", "main"] if case in DEEP else ["--detach"]
        subprocess.run([*git, "checkout", *checkout, base], check=True)
        subprocess.run([*git, "config", "user.email", "qwen@localhost"], check=True)
        subprocess.run([*git, "config", "user.name", "Qwen"], check=True)
        subprocess.run([*container, "python3", "-c", CHECK_RELAY], check=True)
        env = [
            "HOME=/home/evalrun", "USER=evalrun", "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/go/bin",
            "ANTHROPIC_API_KEY=local-eval", "ANTHROPIC_BASE_URL=http://127.0.0.1:30020",
            f"ANTHROPIC_MODEL={MODEL}", f"ANTHROPIC_DEFAULT_HAIKU_MODEL={MODEL}",
            f"ANTHROPIC_DEFAULT_SONNET_MODEL={MODEL}", f"ANTHROPIC_DEFAULT_OPUS_MODEL={MODEL}",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144",
            "DISABLE_AUTOUPDATER=1", "API_TIMEOUT_MS=5400000",
        ]
        tools = "Read,Write,Edit,Bash,Glob,Grep"
        cmd = [
            "docker", "exec", "-w", "/app", name, "runuser", "-u", "evalrun", "--",
            "env", "-i", *env, "timeout", "-k", "10", str(int(deadline)), "claude", "--bare", "-p",
            "--model", MODEL, "--effort", "xhigh", "--permission-mode", "acceptEdits",
            "--allowedTools", tools, "--tools", tools, "--strict-mcp-config", "--no-session-persistence",
            "--output-format", "stream-json", "--verbose", prompt,
        ]
        with (output / "trajectory.jsonl").open("w") as log, (output / "stderr.log").open("w") as err:
            result = subprocess.run(cmd, stdout=log, stderr=err)
        if case in PRO:
            subprocess.run([*git, "add", "-A"], check=True)
            diff = ["diff", "--cached", "--binary", base]
        else:
            diff = ["diff", "--binary", base, "HEAD"]
        patch = subprocess.check_output([*git, *diff])
        (output / "candidate.patch").write_bytes(patch)
        if case in PRO:
            prediction = [{"instance_id": instance, "patch": patch.decode(), "prefix": "qwen"}]
            (output / "prediction.json").write_text(json.dumps(prediction))
        metadata = dict(instance=instance, base=base, cli=version, exit_code=result.returncode,
                        patch_bytes=len(patch), seconds=time.monotonic() - start, model=MODEL, case=case)
        with (output / "trajectory.jsonl").open() as events:
            for line in events:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "result":
                    continue
                usage = (event.get("modelUsage") or {}).get(MODEL, {})
                metadata.update(
                    is_error=event.get("is_error"), turns=event.get("num_turns"),
                    terminal_reason=event.get("subtype"),
                    model_usage={k: usage.get(k) for k in ("inputTokens", "outputTokens", "contextWindow",
                                                           "maxOutputTokens", "cacheReadInputTokens")},
                )
        # Publish the completion marker only after patch and verifier input are available.
        tmp = output / "metadata.tmp"
        tmp.write_text(json.dumps(metadata, indent=2))
        os.replace(tmp, output / "metadata.json")
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[*PRO, *DEEP])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_case(args.case, args.output)
