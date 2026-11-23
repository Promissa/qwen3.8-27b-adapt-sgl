import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from envelope_sweep import classify
from conc_ladder import parse_metrics
from probe import concurrency_activity


class MeasurementClassificationTests(unittest.TestCase):
    def test_pool_is_not_proof_of_memory_failure(self):
        failed = {"status": "failed", "reported_context_capacity": 35000,
                  "error": "JSONDecodeError: bad acknowledgement"}
        self.assertEqual(classify(failed, "max_total_num_tokens=35000"),
                         "infrastructure_or_unclassified_failure")
        self.assertEqual(classify(failed, "torch.OutOfMemoryError: CUDA out of memory"), "memory_failure")
        self.assertEqual(classify(failed,
            "ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.92"),
            "static_budget_failure")
        self.assertEqual(classify({"status": "failed", "error":
            "RuntimeError: Reported capacity 8000 cannot fit 8192/1024"}, ""), "insufficient_token_pool")

    def test_missing_metrics_stay_unknown(self):
        metrics = parse_metrics('sglang:num_running_reqs{model_name="qwen"} 1\n'
                                'sglang:num_retracted_requests_total{model_name="qwen"} 0\n')
        self.assertEqual(metrics["sglang:num_running_reqs"], 1.)
        self.assertEqual(metrics["sglang:num_retracted_requests_total"], 0.)
        self.assertIsNone(metrics["sglang:spec_accept_length"])
        self.assertIsNone(metrics["sglang:full_token_usage"])

    def test_queued_clients_and_single_sample_do_not_prove_concurrency(self):
        for rows in ([], [{"at": 0., "running_requests": None}],
                     [{"at": 0., "running_requests": 1}, {"at": 2., "running_requests": 1}],
                     [{"at": 0., "running_requests": 2}]):
            self.assertFalse(concurrency_activity(rows, 2)["passed"])
        self.assertTrue(concurrency_activity([
            {"at": 0., "running_requests": 2}, {"at": .6, "running_requests": 2}], 2)["passed"])


if __name__ == "__main__":
    unittest.main()
