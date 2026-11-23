#!/usr/bin/env python3
"""SGLang-specific parser checks; record raw responses and explicit failures."""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
import sys
from common import api, now, write_json

VISION_PROMPT = ("Name the color and geometric shape of each large colored object in this image. "
                 "Reply only with color and shape names, separated by commas. "
                 "Do not include coordinates, bounding boxes, markup, or the frame number.")


def contains_description(text, description):
    # Preserve adjacent color/shape words; punctuation does not change their
    # association. In particular, red-circle/blue-square still fails red-square.
    pattern = r"\b" + r"\W+".join(re.escape(word) for word in description.split()) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


class VisionCheckError(AssertionError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def chat(base, prompt, **kw):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    return api(base, "/v1/chat/completions", body)


def vision_check(base, data, expected=()):
    response = chat(base, [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(data).decode()}},
        {"type": "text", "text": VISION_PROMPT}])
    choice = response["choices"][0]
    text = choice["message"].get("content") or ""
    evidence = {"response": response, "image_sha256": hashlib.sha256(data).hexdigest(),
                "expected_descriptions": list(expected), "expected_objects_verified": False,
                "requested_caption_prompt": VISION_PROMPT}
    try:
        assert text, "Empty vision response"
        assert choice["finish_reason"] != "length", "Truncated vision response"
        for description in expected:
            assert contains_description(text, description), f"Missing expected image object: {description}"
    except AssertionError as error:
        raise VisionCheckError(str(error), evidence) from error
    evidence["expected_objects_verified"] = bool(expected)
    return evidence


def cases(base, image=None):
    def tools():
        response = chat(base, "Call get_weather for Hong Kong.", tools=[{"type": "function", "function": {
            "name": "get_weather", "description": "Get city weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
            tool_choice={"type": "function", "function": {"name": "get_weather"}})
        call = response["choices"][0]["message"]["tool_calls"][0]
        assert call["type"] == "function" and call["function"]["name"] == "get_weather"
        assert isinstance(json.loads(call["function"]["arguments"])["city"], str)
        return response

    def thinking(enabled, effort=None, sampled=False):
        options = {"chat_template_kwargs": {"enable_thinking": enabled}, "max_tokens": 2048}
        if sampled:
            options.update(temperature=1., top_p=.95, top_k=20, seed=1701)
        if effort:
            options["reasoning_effort"] = effort
        response = chat(base, "What is 17 times 23? Answer briefly.", **options)
        message = response["choices"][0]["message"]
        assert "391" in (message.get("content") or "")
        if enabled:
            assert message.get("reasoning_content"), "Missing separated reasoning_content"
            assert "<think>" not in (message.get("content") or "")
        else:
            assert not message.get("reasoning_content"), "Thinking disabled but reasoning returned"
        assert response["choices"][0]["finish_reason"] != "length", "Truncated output"
        return response

    checks = [("tool_calls", tools), ("reasoning_content", lambda: thinking(True)),
              ("thinking_disabled", lambda: thinking(False)),
              ("reasoning_effort", lambda: thinking(True, "low")),
              ("thinking_sampled", lambda: thinking(True, sampled=True))]
    if image:
        def vision():
            path = Path(image)
            data = path.read_bytes()
            sidecar = path.with_suffix(".json")
            expected = ()
            if sidecar.exists():
                fixture = json.loads(sidecar.read_text())
                assert hashlib.sha256(data).hexdigest() == fixture["sha256"], "Fixture hash mismatch"
                expected = fixture.get("expected_description", ())
            return vision_check(base, data, expected)
        checks.append(("image_input", vision))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", help="PNG fixture, requires a vision-enabled server")
    args = parser.parse_args()
    results = {"at": now(), "server_info": api(args.base_url, "/get_server_info"), "checks": []}
    for name, fn in cases(args.base_url, args.image):
        try:
            response = fn()
            row = {"name": name, "passed": True, "response": response}
        except Exception as e:
            row = {"name": name, "passed": False, "error": f"{type(e).__name__}: {e}"}
            if hasattr(e, "evidence"):
                row["evidence"] = e.evidence
        results["checks"].append(row)
        write_json(args.output, results)
    return 0 if all(row["passed"] for row in results["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
