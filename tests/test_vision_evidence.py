import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from api_smoke import vision_check, VisionCheckError


class VisionEvidenceTests(unittest.TestCase):
    def test_punctuation_does_not_change_adjacent_color_shape_pairs(self):
        response = {"choices": [{"message": {"content": "red, square, blue, circle"}, "finish_reason": "stop"}]}
        with patch("api_smoke.chat", return_value=response):
            evidence=vision_check("http://unused",b"fixture",("red square","blue circle"))
        self.assertTrue(evidence['expected_objects_verified'])
        response['choices'][0]['message']['content']='red, circle, blue, square'
        with patch("api_smoke.chat",return_value=response), self.assertRaises(VisionCheckError):
            vision_check("http://unused",b"fixture",("red square","blue circle"))

    def test_nonempty_wrong_objects_do_not_pass(self):
        response = {"choices": [{"message": {"content": "A red circle and a blue square."}, "finish_reason": "stop"}]}
        with patch("api_smoke.chat", return_value=response), self.assertRaisesRegex(AssertionError, "Missing expected"):
            vision_check("http://unused", b"fixture", ("red square", "blue circle"))

    def test_failure_retains_raw_response_without_passing_verification(self):
        response = {"choices": [{"message": {"content": "red rectangle, blue circle"}, "finish_reason": "stop"}]}
        with patch("api_smoke.chat", return_value=response), self.assertRaises(VisionCheckError) as caught:
            vision_check("http://unused", b"fixture", ("red square", "blue circle"))
        evidence=caught.exception.evidence
        self.assertEqual(evidence['response'],response)
        self.assertFalse(evidence['expected_objects_verified'])
        self.assertEqual(len(evidence['image_sha256']),64)

    def test_correct_objects_and_complete_response(self):
        response = {"choices": [{"message": {"content": "red square, blue circle"}, "finish_reason": "stop"}]}
        with patch("api_smoke.chat", return_value=response):
            evidence = vision_check("http://unused", b"fixture", ("red square", "blue circle"))
        self.assertTrue(evidence["expected_objects_verified"])
        response["choices"][0]["finish_reason"] = "length"
        with patch("api_smoke.chat", return_value=response), self.assertRaisesRegex(AssertionError, "Truncated"):
            vision_check("http://unused", b"fixture", ("red square", "blue circle"))


if __name__ == "__main__":
    unittest.main()
