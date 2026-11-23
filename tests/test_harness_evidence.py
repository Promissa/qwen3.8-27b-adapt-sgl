"""A passing artifact must not substitute for a real harness tool round trip."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from run_harness import audit_tool_roundtrip


def tool(status="completed"):
    return {"type": "tool_use", "part": {"type": "tool", "id": "call1",
        "tool": "execute", "messageID": "assistant1",
        "state": {"status": status, "output": "tests passed"}}}


def text(message="assistant2"):
    return {"type": "text", "part": {"messageID": message, "text": "Fixed and tested."}}


class HarnessEvidenceTests(unittest.TestCase):
    def audit(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            return audit_tool_roundtrip(path)

    def test_actual_tool_result_then_next_assistant_passes(self):
        self.assertTrue(self.audit([tool(), text()])["passed"])

    def test_claim_without_completed_tool_cannot_pass(self):
        self.assertFalse(self.audit([text()])["passed"])
        self.assertFalse(self.audit([tool("error"), text()])["passed"])

    def test_tool_call_needs_following_assistant_message(self):
        self.assertFalse(self.audit([text(), tool()])["passed"])
        self.assertFalse(self.audit([tool(), text("assistant1")])["passed"])

    def test_missing_result_cannot_pass(self):
        event = tool()
        del event["part"]["state"]["output"]
        self.assertFalse(self.audit([event, text()])["passed"])


if __name__ == "__main__":
    unittest.main()
