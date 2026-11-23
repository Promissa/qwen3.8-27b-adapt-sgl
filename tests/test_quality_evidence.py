"""Reject partial/mismatched scores and apply percentage points correctly."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from compare_quality import compare, read_run


def fixture(path, incorrect=0, cap=16384):
    path.mkdir()
    rows = [{"id": f"gsm8k-{i}", "problem": f"problem{i}", "expected_answer": "1",
        "symbolic_correct": i >= incorrect, "finish_reason": "stop", "num_generated_tokens": 10}
        for i in range(1319)]
    (path / "output-rs0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "metrics.json").write_text(json.dumps({"name": "gsm8k", "num_examples": 1319,
        "n_repeats": 1, "aggregate": {"score": (1319 - incorrect) / 1319},
        "generation": {"max_tokens": cap, "temperature": 1., "seed": 1701}}))


class QualityEvidenceTests(unittest.TestCase):
    def test_six_lost_answers_pass_but_seven_exceed_half_a_point(self):
        with tempfile.TemporaryDirectory() as directory:
            base, six, seven = [Path(directory) / name for name in ("base", "six", "seven")]
            fixture(base); fixture(six, 6); fixture(seven, 7)
            self.assertTrue(compare(base, six)["gsm8k_gate_passed"])
            self.assertFalse(compare(base, seven)["gsm8k_gate_passed"])

    def test_partial_rows_cannot_use_a_full_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"; fixture(path)
            predictions = path / "output-rs0.jsonl"
            predictions.write_text("\n".join(predictions.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "prediction IDs"):
                read_run(path)

    def test_mismatched_budget_and_wrong_aggregate_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base, other = Path(directory) / "base", Path(directory) / "other"
            fixture(base); fixture(other, cap=32768)
            with self.assertRaisesRegex(ValueError, "protocols"):
                compare(base, other)
            metrics = json.loads((base / "metrics.json").read_text())
            metrics["aggregate"]["score"] = .99
            (base / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_run(base)

    def test_silent_output_clamping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"; fixture(path)
            predictions = path / "output-rs0.jsonl"
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            rows[0]["finish_reason"] = "length"
            rows[0]["num_generated_tokens"] = 8192
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Actual output budget"):
                read_run(path)


if __name__ == "__main__":
    unittest.main()
