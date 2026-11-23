"""Load a converted checkpoint through the real fused DFlash weight loader."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "engine/sglang/python"))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PretrainedConfig
from quantize_dflash import convert
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.runtime_context import get_context, get_parallel


class DFlashConversionTests(unittest.TestCase):
    def test_real_fused_loader_and_shared_scales(self):
        config = PretrainedConfig(architectures=["DFlash2DraftModel"], hidden_size=64,
            intermediate_size=128, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=32, vocab_size=256, rms_norm_eps=1e-6,
            hidden_act="silu", max_position_embeddings=64, rope_theta=10000.,
            dflash_config={"block_size": 8, "selector_rank": 16, "selector_top_k": 4,
                           "target_layer_ids": [0], "conv_kernel_size": 2, "conv_group_size": 16})
        backend = get_fused_op_backend()
        set_fused_op_backend(KernelBackend.TORCH)
        try:
            with (get_context().override_server_args(),
                  get_parallel().override(tp_size=1, tp_rank=0, tp_group=None),
                  patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
                  tempfile.TemporaryDirectory() as directory):
                source, output = Path(directory) / "source", Path(directory) / "converted"
                source.mkdir()
                model = DFlash2DraftModel(config)
                tensors = {}
                for name, param in model.named_parameters():
                    if name.endswith("qkv_proj.weight"):
                        prefix = name.removesuffix("qkv_proj.weight")
                        for label, width, magnitude in (("q", 64, .01), ("k", 32, .1), ("v", 32, 1.)):
                            tensors[prefix + label + "_proj.weight"] = torch.linspace(-magnitude, magnitude, width*64).reshape(width,64).bfloat16()
                    elif name.endswith("gate_up_proj.weight"):
                        prefix = name.removesuffix("gate_up_proj.weight")
                        for label, magnitude in (("gate", .02), ("up", .2)):
                            tensors[prefix + label + "_proj.weight"] = torch.linspace(-magnitude, magnitude, 128*64).reshape(128,64).bfloat16()
                    else:
                        tensors[name] = torch.full_like(param, .125, dtype=torch.bfloat16)
                save_file(tensors, source / "model.safetensors")
                (source / "config.json").write_text(config.to_json_string())
                (source / "revision.json").write_text(json.dumps({"complete": True, "revision": "fixture"}))
                report = convert(source, output, rows=16)
                converted_config = json.loads((output / "config.json").read_text())
                quant_config = ModelOptFp4Config.from_config(converted_config["quantization_config"])
                # The production model loader supplies this class-level mapping.
                quant_config.packed_modules_mapping = {}
                loaded = DFlash2DraftModel(config, quant_config=quant_config)
                with safe_open(output / "model.safetensors", framework="pt") as f:
                    loaded.load_weights((key, f.get_tensor(key)) for key in f.keys())
                    for key in ("fc.weight", "candidate_selector.predecessor_codebook",
                                "layers.0.attention_conv.kernel_projection.weight"):
                        torch.testing.assert_close(f.get_tensor(key), tensors[key])
                    qkv = loaded.layers[0].self_attn.qkv_proj
                    torch.testing.assert_close(qkv.weight, torch.cat([
                        f.get_tensor(f"layers.0.self_attn.{part}_proj.weight") for part in ("q", "k", "v")]))
                    self.assertEqual(torch.unique(qkv.weight_scale_2).numel(), 1)
                    self.assertEqual(torch.unique(loaded.layers[0].mlp.gate_up_proj.weight_scale_2).numel(), 1)
                torch.testing.assert_close(loaded.fc.weight, tensors["fc.weight"].to(loaded.fc.weight.dtype))
                self.assertEqual(len(report["tensors"]), 7)
                self.assertFalse(report["gpu_load_verified"])
                self.assertLess(report["checkpoint_file_bytes"], (source / "model.safetensors").stat().st_size)
                with self.assertRaises(ValueError):
                    convert(source, output)
        finally:
            set_fused_op_backend(backend)


if __name__ == "__main__":
    unittest.main()
