"""Load converted MTP weights through the actual Qwen constructor and loader."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "engine/sglang/python")]
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from quantize_mtp import convert
from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptMixedPrecisionConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class MTPConversionTests(unittest.TestCase):
    def test_native_constructor_fused_loader_and_unchanged_backbone(self):
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            full_attention_interval=1, max_position_embeddings=64,
            rope_scaling={"rope_type": "default"}, partial_rotary_factor=1.)
        pp = SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1, rank=0)
        backend = get_fused_op_backend()
        set_fused_op_backend(KernelBackend.TORCH)
        try:
            with (get_context().override_server_args(),
                  get_parallel().override(tp_size=1, tp_rank=0, tp_group=None, pp_group=pp,
                    pp_rank=0, pp_size=1, attn_tp_rank=0, attn_tp_size=1, attn_tp_group=None),
                  patch("sglang.srt.models.qwen3_5.get_stream", return_value=None),
                  patch("sglang.srt.models.qwen3_5._use_mnnvl_cutedsl_fusion", return_value=False),
                  patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
                  tempfile.TemporaryDirectory() as directory):
                source, output = Path(directory) / "source", Path(directory) / "converted"
                source.mkdir()
                model = Qwen3_5ForCausalLMMTP(config)
                tensors = {}
                for name, param in model.named_parameters():
                    if name.endswith("qkv_proj.weight"):
                        for part, height, magnitude in (("q", 128, .01), ("k", 32, .1), ("v", 32, 1.)):
                            tensors[f"mtp.layers.0.self_attn.{part}_proj.weight"] = torch.linspace(
                                -magnitude, magnitude, height * 64).reshape(height, 64).bfloat16()
                    elif name.endswith("gate_up_proj.weight"):
                        for part, magnitude in (("gate", .02), ("up", .2)):
                            tensors[f"mtp.layers.0.mlp.{part}_proj.weight"] = torch.linspace(
                                -magnitude, magnitude, 128 * 64).reshape(128, 64).bfloat16()
                    else:
                        if name == "model.embed_tokens.weight":
                            key = "model.language_model.embed_tokens.weight"
                        elif name == "lm_head.weight":
                            key = name
                        else:
                            key = "mtp." + name.removeprefix("model.")
                            if key.startswith("mtp.layers.0.") and any(
                                    field in key for field in ("o_proj", "q_norm", "k_norm")):
                                key = key.replace("mtp.layers.0.", "mtp.layers.0.self_attn.")
                        tensors[key] = torch.full_like(param, .125, dtype=torch.bfloat16)
                original = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                    "ignore": ["mtp*", "mtp.layers.0*"], "quantized_layers": {
                        "backbone": {"quant_algo": "NVFP4", "group_size": 16}}}
                (source / "config.json").write_text(json.dumps({"text_config": config.to_dict(),
                    "quantization_config": original}))
                (source / "revision.json").write_text(json.dumps({"complete": True, "revision": "fixture"}))
                save_file(tensors, source / "model-00001-of-00002.safetensors")
                save_file({"backbone.weight": torch.arange(128, dtype=torch.uint8).reshape(8, 16)},
                          source / "model-00002-of-00002.safetensors")
                report = convert(source, output, rows=16)
                changed = json.loads((output / "config.json").read_text())["quantization_config"]
                quant = ModelOptMixedPrecisionConfig.from_config(changed)
                quant.packed_modules_mapping = model.packed_modules_mapping
                self.assertIs(_mtp_quant_config(quant), quant)
                loaded = Qwen3_5ForCausalLMMTP(config, quant_config=quant)
                with safe_open(output / "model-00001-of-00002.safetensors", framework="pt") as handle:
                    loaded.load_weights((key, handle.get_tensor(key)) for key in handle.keys())
                    qkv = loaded.model.layers[0].qkv_proj
                    torch.testing.assert_close(qkv.weight, torch.cat([handle.get_tensor(
                        f"mtp.layers.0.self_attn.{part}_proj.weight") for part in ("q", "k", "v")]))
                    self.assertEqual(torch.unique(qkv.weight_scale_2).numel(), 1)
                    self.assertEqual(torch.unique(loaded.model.layers[0].mlp.gate_up_proj.weight_scale_2).numel(), 1)
                torch.testing.assert_close(loaded.fc.weight, tensors["mtp.fc.weight"].to(loaded.fc.weight.dtype))
                self.assertTrue((output / "model-00002-of-00002.safetensors").is_symlink())
                self.assertEqual(changed["quantized_layers"]["backbone"], original["quantized_layers"]["backbone"])
                self.assertEqual(json.loads((source / "config.json").read_text())["quantization_config"], original)
                self.assertEqual(len(report["tensors"]), 7)
                self.assertLess(report["converted_mtp_tensor_bytes"], report["source_mtp_tensor_bytes"])
                self.assertFalse(report["gpu_load_verified"])
                # Exercise actual converted-module ownership: share the entire
                # quantized head (including scales), while retaining draft MTP.
                draft = Qwen3_5ForCausalLMMTP(config, quant_config=quant)
                before = {name: id(param) for name, param in draft.named_parameters()
                          if not name.startswith(("model.embed_tokens.", "lm_head."))}
                head_quant = ModelOptFp4Config.from_config({
                    "quant_method": "modelopt_fp4", "quant_algo": "NVFP4",
                    "group_size": 16, "exclude_modules": [], "packed_modules_mapping": {}})
                loaded.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                               quant_config=head_quant, prefix="lm_head")
                self.assertEqual(loaded.lm_head.weight.dtype, torch.uint8)
                self.assertTrue(hasattr(loaded.lm_head, "weight_scale"))
                worker = EagleDraftWorker.__new__(EagleDraftWorker)
                worker.draft_runner = SimpleNamespace(model=draft)
                worker.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=loaded))
                worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
                module = "sglang.srt.speculative.eagle_worker_v2"
                with (patch(f"{module}.envs.SGLANG_OPT_QWEN_MTP_EARLY_SHARE.get", return_value=True),
                      patch(f"{module}.get_model", return_value=SimpleNamespace(is_startup_weight_load_overlap=False)),
                      patch(f"{module}.get_spec", return_value=SimpleNamespace(speculative_token_map=None)),
                      patch.object(worker, "_resolve_shared_embed_and_head", return_value=loaded.get_embed_and_head())):
                    worker._maybe_share_qwen_mtp_before_pool()
                self.assertIs(draft.model.embed_tokens.weight, loaded.model.embed_tokens.weight)
                self.assertIs(draft.lm_head, loaded.lm_head)
                self.assertTrue(worker._shared_weights_initialized)
                self.assertEqual(before, {name: id(param) for name, param in draft.named_parameters()
                                 if not name.startswith(("model.embed_tokens.", "lm_head."))})
                # A packed target needs the gather implementation and both
                # scales. Sharing only its uint8 weight breaks MTP lookup.
                embed_prefix = "model.language_model.embed_tokens"
                embed_quant = ModelOptMixedPrecisionConfig.from_config({
                    "quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION",
                    "group_size": 16, "exclude_modules": [], "packed_modules_mapping": {},
                    "quantized_layers": {embed_prefix: {"quant_algo": "NVFP4", "group_size": 16}},
                })
                packed = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                    params_dtype=draft.model.embed_tokens.weight.dtype,
                    quant_config=embed_quant, prefix=embed_prefix)
                loaded.model.embed_tokens = packed
                with (patch.object(loaded, "get_input_embeddings", return_value=packed, create=True),
                      patch(f"{module}.envs.SGLANG_OPT_QWEN_MTP_EARLY_SHARE.get", return_value=True),
                      patch(f"{module}.get_model", return_value=SimpleNamespace(is_startup_weight_load_overlap=False)),
                      patch(f"{module}.get_spec", return_value=SimpleNamespace(speculative_token_map=None)),
                      patch.object(worker, "_resolve_shared_embed_and_head", return_value=loaded.get_embed_and_head())):
                    with patch.object(packed, "embedding_dim", config.hidden_size + 16):
                        with self.assertRaisesRegex(ValueError, "Incompatible NVFP4 embedding"):
                            worker._maybe_share_qwen_mtp_before_pool()
                    worker._maybe_share_qwen_mtp_before_pool()
                self.assertIs(draft.model.embed_tokens, packed)
                self.assertIs(draft.lm_head, loaded.lm_head)
                with self.assertRaises(ValueError):
                    convert(source, output)
        finally:
            set_fused_op_backend(backend)

    def test_early_sharing_default_off_and_unsupported_model(self):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        module = "sglang.srt.speculative.eagle_worker_v2"
        with patch(f"{module}.envs.SGLANG_OPT_QWEN_MTP_EARLY_SHARE.get", return_value=False):
            worker._maybe_share_qwen_mtp_before_pool()
        self.assertFalse(worker._shared_weights_initialized)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.draft_runner = SimpleNamespace(model=torch.nn.Linear(4, 4))
        with patch(f"{module}.envs.SGLANG_OPT_QWEN_MTP_EARLY_SHARE.get", return_value=True):
            with self.assertRaisesRegex(ValueError, "Early Qwen MTP sharing requires"):
                worker._maybe_share_qwen_mtp_before_pool()


if __name__ == "__main__":
    unittest.main()
