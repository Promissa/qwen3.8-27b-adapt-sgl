"""Cache claims require matching controls, output IDs and actual cache hits."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from prefix_dialogue import checked_stream, gates


def dialogue():
    turns = []
    for index in range(2):
        response = {"output_ids": [1, 2]}
        warm = {"retrieval_passed": True, "cached_tokens": 100 if index else 0,
                "ttft_s": .2, "response": response}
        cold = copy.deepcopy(warm); cold["cached_tokens"] = 0
        turns.append({"round": index, "warm": warm, "cold": cold})
    return [{"turns": turns}]


class PrefixEvidenceTests(unittest.TestCase):
    def test_positive_controls_and_cache_hits_pass(self):
        self.assertTrue(all(gates(dialogue()).values()))

    def test_unflushed_control_and_no_hit_cannot_pass(self):
        rows = dialogue(); rows[0]["turns"][1]["cold"]["cached_tokens"] = 1
        self.assertFalse(gates(rows)["flushed_control_gate"])
        rows = dialogue(); rows[0]["turns"][0]["warm"]["cached_tokens"] = 1
        self.assertFalse(gates(rows)["cold_bootstrap_gate"])
        rows = dialogue(); rows[0]["turns"][1]["warm"]["cached_tokens"] = 0
        self.assertFalse(gates(rows)["followup_cache_hit_gate"])
        self.assertFalse(gates(rows)["followup_ttft_1_5s_gate"])

    def test_same_answer_with_different_tokens_or_slow_hit_is_recorded(self):
        rows = dialogue(); rows[0]["turns"][1]["cold"]["response"]["output_ids"] = [3, 4]
        self.assertFalse(gates(rows)["cache_token_equality_gate"])
        rows = dialogue(); rows[0]["turns"][1]["warm"]["ttft_s"] = 1.50001
        self.assertFalse(gates(rows)["followup_ttft_1_5s_gate"])

    def test_request_length_or_missing_cache_evidence_is_rejected(self):
        response = {"ttft_s": .1, "response": {"text": "answer", "output_ids": [1],
            "meta_info": {"prompt_tokens": 2, "completion_tokens": 1,
                "finish_reason": {"type": "stop"}}}}
        with patch("prefix_dialogue.stream", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "cache-token"):
                checked_stream("http://unused", [1, 2], "answer", 1000)
            with self.assertRaisesRegex(RuntimeError, "prompt/output"):
                checked_stream("http://unused", [1], "answer", 1000)


if __name__ == "__main__":
    unittest.main()
