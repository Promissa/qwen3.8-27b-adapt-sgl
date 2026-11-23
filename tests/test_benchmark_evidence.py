"""Reject incomplete or truncated evidence before producing a performance claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from summarize_serving import summarize, validate


def record():
    return {"backend": "sglang-oai", "dataset_name": "random", "max_concurrency": 1,
            "random_range_ratio": 1, "random_input_len": 1024, "random_output_len": 1024,
            "input_lens": [1024, 1024], "output_lens": [1024, 1024], "errors": ["", ""],
            "completed": 2, "total_input_tokens": 2048, "total_output_tokens": 2048,
            "output_throughput": 70., "median_ttft_ms": 75., "median_tpot_ms": 14.3}


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_truncated_outputs_are_rejected_even_with_matching_aggregates(self):
        row = record()
        validate(row, 1024)
        row["output_lens"][1] = 20
        row["total_output_tokens"] = 1044
        with self.assertRaisesRegex(ValueError, "Actual input/output"):
            validate(row, 1024)

    def test_missing_repeat_cannot_be_a_complete_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for repetition in (1, 2):
                (folder / f"isl1024-rep{repetition}.jsonl").write_text(json.dumps(record()) + "\n")
            report = summarize(folder)
            self.assertEqual(report["status"], "incomplete")
            self.assertNotIn("metrics", report["groups"][0])
            self.assertEqual(len(report["missing"]), 4)


if __name__ == "__main__":
    unittest.main()
