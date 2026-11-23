"""Local Claude Code Messages adapter for the evaluated Qwen3.8-27B profile."""

import argparse
import json
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web


async def proxy(request):
    body = await request.json()
    event = {
        "path": request.path,
        "model": body.get("model"),
        "tools": [tool.get("name") for tool in body.get("tools", [])],
        "tool_return": any(
            isinstance(message.get("content"), list)
            and any(block.get("type") == "tool_result" for block in message["content"])
            for message in body.get("messages", [])
        ),
    }
    if body.get("model") == "qwen3.8-27b" and request.path == "/v1/messages":
        body.update(temperature=1.0, top_p=0.95)
        event.update(temperature_sent=1.0, top_p_sent=0.95)
        config = body.get("output_config", {})
        event["incoming_effort"] = config.get("effort")
        if config.get("effort") in {"high", "xhigh", "max"}:
            # SGLang's Anthropic parser normalizes explicit xhigh to max.
            # Omitting the field uses the checkpoint template's native xhigh.
            config.pop("effort")
            event["qwen_effort"] = "template default xhigh"

    # A full 32K decode can take close to an hour on the single-card offload profile.
    # The benchmark harness still owns the per-task wall-clock deadline.
    timeout = ClientTimeout(total=None, sock_connect=10, sock_read=5400)
    async with ClientSession(timeout=timeout) as client:
        async with client.post(request.app["upstream"] + request.path, json=body) as upstream:
            event["status"] = upstream.status
            with request.app["audit"].open("a") as log:
                print(json.dumps(event), file=log)
            response = web.StreamResponse(
                status=upstream.status,
                headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")},
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--upstream-port", type=int, default=30019)
    parser.add_argument("--port", type=int, default=30020)
    args = parser.parse_args()
    app = web.Application(client_max_size=32 * 1024**2)
    app["upstream"] = f"http://127.0.0.1:{args.upstream_port}"
    app["audit"] = args.audit
    for endpoint in ("/v1/messages", "/v1/messages/count_tokens"):
        app.router.add_post(endpoint, proxy)
    web.run_app(app, host="127.0.0.1", port=args.port)
