"""Regression checks for the actual Qwen3.5 CLI gate in the isolated engine."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine/sglang/python"
if ENGINE.exists():
    sys.path.insert(0, str(ENGINE))

from sglang.srt.arg_groups.model_hook import handle_language_model_only
from sglang.srt.server_args import ServerArgs


class LanguageModelOnlyTests(unittest.TestCase):
    def check_arch(self, arch, **kwargs):
        args = ServerArgs(model_path=str(ROOT / "weights/target"), language_model_only=True, **kwargs)
        config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch], model_type="qwen3_5"))
        with patch("sglang.srt.arg_groups.model_hook.model_config_of", return_value=config):
            handle_language_model_only(args)

    def test_qwen35_text_only_is_accepted(self):
        self.check_arch("Qwen3_5ForConditionalGeneration")

    def test_unverified_architecture_stays_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.check_arch("UnverifiedForConditionalGeneration")

    def test_multimodal_encoder_conflict_stays_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.check_arch("Qwen3_5ForConditionalGeneration", encoder_only=True)

    def test_pd_disaggregation_stays_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.check_arch("Qwen3_5ForConditionalGeneration", disaggregation_mode="prefill")


if __name__ == "__main__":
    unittest.main()
