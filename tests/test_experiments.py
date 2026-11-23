"""Offline checks for measurement integrity, process cleanup, and quantized layout."""
import argparse
import ast
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "bench")]
import envelope_sweep
import probe
import serve
import token_compare
import common
from quantize_embedding import convert, quantize_rows, update_config
import torch
from safetensors import safe_open
from safetensors.torch import save_file


class MeasurementTests(unittest.TestCase):
    def test_effective_checkpoint_override_and_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text("{}")
            (directory / "revision.json").write_text('{"complete": false}')
            launch = ["python", "--model-path", "/unused/default", "--model-path", str(directory)]
            self.assertEqual(serve.effective_argument(launch, "--model-path"), str(directory))
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                serve.validate_checkpoints(launch)

    def test_plain_text_control_acknowledgement_but_strict_generation_json(self):
        reply = SimpleNamespace(read=lambda: b"Cache flushed successfully")
        with patch.object(common.urllib.request, "urlopen", return_value=nullcontext(reply)):
            self.assertEqual(common.api("http://unused", "/flush_cache", {}), "Cache flushed successfully")
        with patch.object(common.urllib.request, "urlopen", return_value=nullcontext(reply)):
            with self.assertRaises(json.JSONDecodeError):
                common.api("http://unused", "/generate", {})

    def test_bracket_discovers_middle_passes_and_high_boundary(self):
        best, samples = envelope_sweep.bracket_search(lambda x: .91 <= x <= .955, .88, .98, .001)
        self.assertLessEqual(best, .955)
        self.assertGreater(best, .954)
        self.assertFalse(samples[.88])
        self.assertFalse(samples[.98])

    def test_no_pass_does_not_report_capacity(self):
        best, _ = envelope_sweep.bracket_search(lambda x: False, .88, .98, .005)
        self.assertIsNone(best)

    def test_capacity_honors_model_context_limit(self):
        info = {"internal_states": [{"max_total_num_tokens": 65536}],
                "server_args": {"context_length": 16384}}
        self.assertEqual(probe.capacity(info, {}), 16384)
        with self.assertRaises(RuntimeError):
            probe.capacity({}, {})

    def test_log_uses_final_post_capture_token_count(self):
        parsed = probe.parse_log("max_total_num_tokens=4096\nPost-capture max_total_num_tokens=8192\n"
                                 "max_mamba_cache_size=4 max_running_requests=1")
        self.assertEqual(parsed["max_total_num_tokens"], 8192)
        self.assertEqual(parsed["max_mamba_cache_size"], 4)

    def test_truncated_request_cannot_pass(self):
        with patch.object(probe, "api", return_value={"meta_info": {
            "prompt_tokens": 8192, "completion_tokens": 2}}):
            with self.assertRaisesRegex(RuntimeError, "Truncated"):
                probe.generate("http://unused", 8192, 1024)

    def test_token_comparison_uses_ids_and_protocol(self):
        a = {"complete": True, "protocol": {"hash": "same", "prompt_count": 1}, "outputs": [{"output_ids": [1, 2], "text": "same"}]}
        b = copy.deepcopy(a)
        b["outputs"][0]["output_ids"] = [3, 4]
        self.assertFalse(token_compare.compare(a, b))
        b["protocol"]["hash"] = "different"
        with self.assertRaises(ValueError):
            token_compare.compare(a, b)
        b = copy.deepcopy(a)
        b["complete"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            token_compare.compare(a, b)

    def test_probe_owns_and_cleans_real_child_on_request_failure(self):
        # No sockets/GPU are available in this sandbox. Exercise the real process
        # lifetime, with only the HTTP and nvidia-smi observations substituted.
        created = []
        real_popen = subprocess.Popen

        def popen(*args, **kw):
            child = real_popen(*args, **kw)
            # Provenance also launches short read-only git commands. This
            # lifetime invariant concerns the owned server process group.
            if kw.get("start_new_session"):
                created.append(child)
            return child

        class Socket:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def setsockopt(self, *args): pass
            def bind(self, address): pass

        def api(base, route, *args, **kw):
            if route == "/get_server_info":
                return {"max_total_num_tokens": 16384}
            if route == "/generate":
                raise RuntimeError("simulated prefill OOM")
            return None

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), desktop="unknown", concurrency=1,
                                   port=30000, ready_timeout=10, profile="balanced")
            with patch("socket.socket", return_value=Socket()), \
                 patch.object(probe.subprocess, "Popen", side_effect=popen), \
                 patch.object(probe.urllib.request, "urlopen", return_value=nullcontext(SimpleNamespace(status=200))), \
                 patch.object(probe, "api", side_effect=api), \
                 patch.object(probe, "gpu_sample", return_value={"gpus": [{"used_mib": 123}]}):
                status = probe.run(args, [sys.executable, "-c", "import time; time.sleep(60)"])
            self.assertEqual(status, 1)
            self.assertEqual(len(created), 1)
            self.assertIsNotNone(created[0].poll())
            report = json.loads((Path(tmp) / "probe.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("prefill OOM", report["error"])

    def test_profile_options_exist_in_current_server_help(self):
        help_path = ROOT / "results/server-help.txt"
        if not help_path.exists():
            self.skipTest("Run setup first to obtain server-help.txt")
        available = help_path.read_text()
        for profile in serve.PROFILES:
            with patch.dict("os.environ", {"MODEL_PATH": "/fixture/model"}):
                cmd = serve.command(profile)
            for option in [arg for arg in cmd if arg.startswith("--")]:
                self.assertIn(option, available, (profile, option))
        experiments = json.loads((ROOT / "configs/experiments.json").read_text())
        for experiment in experiments["experiments"]:
            for option in experiment["flags"]:
                if option.startswith("--"):
                    self.assertIn(option, available, experiment["name"])


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def upstream_gather(self, packed, scales, scale2):
        # Execute the actual audited source method, avoiding GPU-only module imports.
        source = Path.home() / "projects/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py"
        if not source.exists():
            self.skipTest("Audited SGLang source unavailable")
        tree = ast.parse(source.read_text())
        embedding = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                         and n.name == "ModelOptNvFp4EmbeddingMethod")
        method = next(n for n in embedding.body if isinstance(n, ast.FunctionDef) and n.name == "embedding")
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                           str(source), "exec")
        namespace = {"torch": torch}
        exec(compiled, namespace)
        layer = SimpleNamespace(weight=packed, weight_scale=scales,
            weight_scale_2=torch.tensor([scale2]), e2m1_lut=torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]))
        config = SimpleNamespace(quant_config=SimpleNamespace(group_size=16), params_dtype=torch.float32)
        return namespace["embedding"](config, layer, torch.arange(packed.shape[0]))

    def test_exact_e2m1_values_and_nibble_order_against_upstream_gather(self):
        values = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6., 0., -.5, -1., -1.5, -2., -3., -4., -6.]])
        packed, scales = quantize_rows(values, 1.)
        self.assertEqual(packed.tolist(), [[16, 50, 84, 118, 144, 186, 220, 254]])
        self.assertTrue(torch.equal(self.upstream_gather(packed, scales, 1.), values))

    def test_zero_and_extreme_blocks_are_finite(self):
        values = torch.cat([torch.zeros(1, 16), torch.full((1, 16), 1000.)], dim=0)
        scale2 = 1000 / (448 * 6)
        packed, scales = quantize_rows(values, scale2)
        result = self.upstream_gather(packed, scales, scale2)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[0], values[0]))
        torch.testing.assert_close(result[1], values[1], rtol=1e-6, atol=.001)

    def test_sharded_conversion_preserves_other_tensors_and_maps_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / "source", root / "candidate"
            source.mkdir()
            config = {"tie_word_embeddings": False, "quantization_config": {
                "quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["*.embed_tokens", "visual.*"]}}
            (source / "config.json").write_text(json.dumps(config))
            (source / "revision.json").write_text(json.dumps({"revision": "fixture", "complete": True}))
            torch.manual_seed(0)
            embedding = torch.randn(17, 32).bfloat16()
            norm = torch.arange(32).bfloat16()
            save_file({"model.embed_tokens.weight": embedding, "model.norm.weight": norm},
                      source / "model-00001-of-00002.safetensors")
            save_file({"model.layers.0.proj.weight": torch.ones(32, 16, dtype=torch.uint8),
                       "model.layers.0.proj.weight_scale": torch.ones(32, 2).to(torch.float8_e4m3fn),
                       "model.layers.0.proj.weight_scale_2": torch.ones(1)},
                      source / "model-00002-of-00002.safetensors")
            report = convert(source, dest, rows=3)
            self.assertFalse(report["quality_verified"])
            self.assertLess(report["relative_rmse"], .15)
            self.assertFalse((dest / ".conversion-in-progress").exists())
            self.assertTrue((dest / "model-00002-of-00002.safetensors").is_symlink())
            with safe_open(dest / "model-00001-of-00002.safetensors", framework="pt") as f:
                self.assertTrue(torch.equal(f.get_tensor("model.norm.weight"), norm))
                self.assertEqual(f.get_tensor("model.embed_tokens.weight").dtype, torch.uint8)
                result = self.upstream_gather(f.get_tensor("model.embed_tokens.weight"),
                    f.get_tensor("model.embed_tokens.weight_scale"), report["scale_2"])
                self.assertEqual(tuple(result.shape), (17, 32))
            changed = json.loads((dest / "config.json").read_text())
            self.assertEqual(changed["quantization_config"]["quant_method"], "modelopt_mixed")
            self.assertEqual(changed["quantization_config"]["ignore"], ["visual.*"])
            index = json.loads((dest / "model.safetensors.index.json").read_text())
            self.assertIn("model.embed_tokens.weight_scale_2", index["weight_map"])
            # Source is untouched.
            self.assertEqual(json.loads((source / "config.json").read_text()), config)
            with self.assertRaisesRegex(ValueError, "Destination"):
                convert(source, dest)

    def test_reject_tied_embeddings_and_ambiguous_exclusions(self):
        with self.assertRaisesRegex(ValueError, "Tied"):
            update_config({"tie_word_embeddings": True}, "model.embed_tokens", {})
        config = {"tie_word_embeddings": False, "quantization_config": {"quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION", "quantized_layers": {"p": {"quant_algo": "NVFP4"}},
            "ignore": ["*"]}}
        with self.assertRaisesRegex(ValueError, "Broad exclusion"):
            update_config(config, "model.embed_tokens", {})


if __name__ == "__main__":
    unittest.main()
