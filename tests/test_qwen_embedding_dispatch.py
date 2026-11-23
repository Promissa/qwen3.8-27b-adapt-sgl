"""Exercise the real Qwen backbone constructor and embedding weight loader."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/sglang/python"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from quantize_embedding import quantize_rows
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config, ModelOptMixedPrecisionConfig, ModelOptNvFp4EmbeddingMethod,
)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_context, get_parallel


class QwenEmbeddingDispatchTests(unittest.TestCase):
    def build(self, quant_config, first=True):
        config = SimpleNamespace(vocab_size=64, hidden_size=16, num_hidden_layers=0,
                                 layers_block_type=[], rms_norm_eps=1e-6)
        pp = SimpleNamespace(is_first_rank=first, is_last_rank=True, world_size=1)
        with (get_context().override_server_args(),
              get_parallel().override(tp_size=1, tp_rank=0, tp_group=None, pp_group=pp),
              patch("sglang.srt.models.qwen3_5.get_stream", return_value=None),
              patch("sglang.srt.models.qwen3_5.make_pp_layers", return_value=(torch.nn.ModuleList(), 0, 0)),
              patch("sglang.srt.models.qwen3_5._use_mnnvl_cutedsl_fusion", return_value=False)):
            model = Qwen3_5ForCausalLM(config, quant_config, prefix="model.language_model")
        return model, config

    def mixed(self):
        quant = ModelOptMixedPrecisionConfig.from_config({"quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION", "group_size": 16, "exclude_modules": [],
            "packed_modules_mapping": {}, "quantized_layers": {
                "model.language_model.embed_tokens": {"quant_algo": "NVFP4", "group_size": 16}}})
        return quant

    def test_constructor_selects_and_loads_packed_embedding(self):
        model, _ = self.build(self.mixed())
        layer = model.embed_tokens
        self.assertIsInstance(layer.quant_method, ModelOptNvFp4EmbeddingMethod)
        self.assertEqual(layer.weight.dtype, torch.uint8)
        original = torch.linspace(-1, 1, 64*16).reshape(64, 16)
        global_scale = original.abs().max().item()/(6*448)
        packed, scale = quantize_rows(original, global_scale)
        for name, weight in (("weight", packed), ("weight_scale", scale),
                             ("weight_scale_2", torch.tensor([global_scale]))):
            param = getattr(layer, name)
            param.weight_loader(param, weight)
        ids = torch.tensor([0, 31, 63])
        with get_context().override_server_args(), get_parallel().override(tp_size=1, tp_rank=0, tp_group=None):
            output = layer(ids)
        self.assertEqual(tuple(output.shape), (3, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess((output.float()-original[ids]).square().mean().sqrt().item(), .15)

    def test_ordinary_embedding_and_pipeline_guard(self):
        for quant in (None, ModelOptFp4Config.from_config({"quant_method": "modelopt_fp4",
                "quant_algo": "NVFP4", "group_size": 16, "exclude_modules": [], "packed_modules_mapping": {}})):
            model, _ = self.build(quant)
            self.assertTrue(model.embed_tokens.weight.is_floating_point())
            self.assertEqual(model.embed_tokens.weight.shape[1], 16)
        model, _ = self.build(self.mixed(), first=False)
        self.assertIsInstance(model.embed_tokens, PPMissingLayer)

    def test_existing_qwen4_hook_accepts_constructor_contract(self):
        model, config = self.build(self.mixed())
        inherited = Qwen4ExpModel.__new__(Qwen4ExpModel)
        torch.nn.Module.__init__(inherited)
        inherited.pp_group = model.pp_group
        with get_context().override_server_args(), get_parallel().override(tp_size=1, tp_rank=0, tp_group=None):
            layer = inherited._build_embed_tokens(config, quant_config=self.mixed(),
                                                 prefix="model.language_model.embed_tokens")
        self.assertIsInstance(layer.quant_method, ModelOptNvFp4EmbeddingMethod)


if __name__ == "__main__":
    unittest.main()
