"""Run the full single-turn IFBench release against the local combined server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=30019)
    args = parser.parse_args()
    rows = read_rows(args.dataset)
    assert len(rows) == len({r["prompt"] for r in rows}) == len({r["key"] for r in rows}) == 300
    endpoint = f"http://127.0.0.1:{args.port}"
    sampling = dict(model="qwen3.8-27b", temperature=1.0, top_p=0.95, top_k=20,
                    min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
                    max_tokens=4096, seed=42, chat_template_kwargs={"enable_thinking": True})
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = args.directory / "ifbench-protocol.json"
    protocol = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "sampling": sampling, "endpoint": endpoint, "samples_per_prompt": 1}
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Existing run uses a different protocol")
    manifest.write_text(json.dumps(protocol, indent=2) + "\n")
    with urlopen(endpoint + "/server_info", timeout=30) as reply:
        (args.directory / "ifbench-server.json").write_bytes(reply.read())
    output = args.directory / "ifbench-responses.jsonl"
    saved = read_rows(output) if output.exists() else []
    for idx, rec in enumerate(saved):
        if rec["key"] != rows[idx]["key"] or rec["prompt"] != rows[idx]["prompt"]:
            raise ValueError(f"Response order or dataset changed at {idx}")
    with output.open("a") as writer:
        for idx in range(len(saved), len(rows)):
            row = rows[idx]
            payload = dict(sampling, messages=[{"role": "user", "content": row["prompt"]}])
            request = Request(endpoint + "/v1/chat/completions", json.dumps(payload).encode(),
                              {"Content-Type": "application/json"})
            start = time.monotonic()
            with urlopen(request, timeout=1200) as response:
                response = json.load(response)
            choice = response["choices"][0]
            if choice["finish_reason"] not in ("stop", "length"):
                raise ValueError(f"Unexpected finish reason for {row['key']}: {choice['finish_reason']}")
            answer = choice["message"]["content"] or ""
            result = {"key": row["key"], "prompt": row["prompt"], "response": answer,
                      "usage": response["usage"], "finish_reason": choice["finish_reason"],
                      "seconds": round(time.monotonic() - start, 3)}
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
            print(f"{idx + 1}/300 {choice['finish_reason']} {result['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
