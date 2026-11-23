import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from soak import soak_evidence


class SoakEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.required = {"short", "long", "tool_calls", "image_input"}
        self.result = {"requests": [{"name": name} for name in self.required],
                       "elapsed_s": 7201., "requested_seconds": 7200., "vision_included": True}
        self.telemetry = [{"elapsed_s": float(t), "gpus": [{"used_mib": 22000,
            "free_mib": 2000, "utilization_percent": 90}], "host_meminfo": "MemAvailable: 100000 kB\n"}
            for t in range(0, 7201, 10)]

    def test_complete_curve_and_cases_pass(self):
        self.assertTrue(soak_evidence(self.result, self.required, self.telemetry)["full_plan_soak_gate"])

    def test_missing_case_is_a_failed_run(self):
        self.result["requests"] = [{"name": "long"}]
        result = soak_evidence(self.result, self.required, self.telemetry)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["full_plan_soak_gate"])

    def test_runtime_completion_does_not_turn_bad_vision_into_quality_pass(self):
        self.result['requests']=[r for r in self.result['requests'] if r['name']!='image_input']
        self.result.update(vision_quality_errors=[{'name':'image_input'}],
            vision_quality_failure_only=True,error='1 image responses failed object checks')
        result=soak_evidence(self.result,self.required,self.telemetry)
        self.assertTrue(result['runtime_stability_7200s_gate'])
        self.assertFalse(result['full_plan_soak_gate'])
        self.assertFalse(result['all_cases_completed'])
        self.assertNotIn('image_input',result['completed_case_counts'])
        self.assertEqual(result['attempted_case_counts']['image_input'],1)
        self.assertEqual(result['status'],'completed_with_vision_quality_errors')
        self.result['runtime_error']='CUDA OOM'
        self.assertFalse(soak_evidence(self.result,self.required,self.telemetry)['runtime_stability_7200s_gate'])

    def test_absent_placeholder_and_broken_curve_fail(self):
        broken = copy.deepcopy(self.telemetry)
        broken[20:40] = []
        for rows in ([], [{"elapsed_s": 0., "gpus": [{}], "host_meminfo": "MemAvailable:"}], broken):
            self.assertFalse(soak_evidence(self.result, self.required, rows)["full_plan_soak_gate"])
        self.result["monitor_errors"] = ["disk write failed"]
        self.assertFalse(soak_evidence(self.result, self.required, self.telemetry)["full_plan_soak_gate"])

    def test_text_only_short_duration_and_request_error_do_not_pass_full_gate(self):
        self.result["vision_included"] = False
        self.assertFalse(soak_evidence(self.result, self.required, self.telemetry)["full_plan_soak_gate"])
        self.result["vision_included"] = True
        self.result["elapsed_s"] = 7199.
        self.assertEqual(soak_evidence(self.result, self.required, self.telemetry)["status"], "failed")
        self.result["elapsed_s"] = 7201.
        self.result["error"] = "CUDA OOM"
        self.assertEqual(soak_evidence(self.result, self.required, self.telemetry)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
