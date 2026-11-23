"""Use actual PyTorch hooks to check bounded capture and negative comparisons."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from forward_trace import Recorder
from compare_forward_trace import compare, read

spec = importlib.util.spec_from_file_location('actual_sglang_hook_manager', ROOT /
    'engine/sglang/python/sglang/srt/model_executor/hook_manager.py')
actual_hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual_hooks)


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == 'DECODE'

    def is_target_verify(self):
        return self.name == 'TARGET_VERIFY'


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.mlp = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(4))

    def forward(self, hidden_states):
        if hasattr(self, 'linear_attn'):
            hidden_states = self.linear_attn(hidden_states)
        return self.mlp(self.input_layernorm(hidden_states))


class Toy(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(), Layer()])
        self.offset = offset

    def forward(self, input_ids, positions, forward_batch):
        x = input_ids.float()[:, None].expand(-1, 4).clone()
        if forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_target_verify():
            x += self.offset
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(next_token_logits=x)


class Outer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, positions, forward_batch):
        return self.model(input_ids=ids, positions=positions, forward_batch=forward_batch)


class UpdatingGDN(nn.Module):
    layer_id = 0

    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def forward(self, hidden_states):
        self.cache.conv[0][1].add_(1)
        self.cache.temporal[1].add_(1)
        return hidden_states


class OpaqueScaleMLP(nn.Module):
    def __init__(self, scale, dtype):
        super().__init__()
        self.down_proj = nn.Identity()
        self.scale = scale
        self.dtype = dtype

    def forward(self, hidden_states):
        # Match the runtime's packed activation/interleaved scale boundary.
        self.down_proj((torch.zeros(len(hidden_states), 2, dtype=torch.uint8),
                        torch.full((128, 16), self.scale, dtype=self.dtype)))
        return hidden_states


def batch(ids, mode):
    return SimpleNamespace(input_ids=torch.tensor(ids), seq_lens=torch.tensor([len(ids)]),
                           forward_mode=Mode(mode), batch_size=1)


class ForwardTraceTests(unittest.TestCase):
    def capture(self, directory, offset=0, verify=False, limit=1024**2, cache=None, opaque_scale=None):
        backbone = Toy(offset)
        if opaque_scale is not None:
            backbone.model.layers[0].mlp = OpaqueScaleMLP(*opaque_scale)
        if cache is not None:
            backbone.model.layers[0].linear_attn = UpdatingGDN(cache)
        model = Outer(backbone)
        arm = directory / 'armed'
        config = {'output_dir': str(directory / 'trace'), 'arm_file': str(arm),
                  'expected_model_class': 'Toy', 'max_bytes': limit}
        actual_hooks.register_forward_hooks(model, [{'target_modules': ['model'],
            'hook_factory': 'forward_trace:make_hook', 'config': config}])
        recorder = next(iter(backbone._forward_hooks.values())).__self__
        first = batch([1, 2, 3], 'EXTEND')
        # Match EagerRunner's direct outer .forward, which skips outer hooks.
        model.forward(first.input_ids, torch.arange(3), first)
        model.forward(first.input_ids, torch.arange(3), first)
        self.assertEqual(recorder.state['frames'], [])
        self.assertFalse(list(recorder.output.glob('*.pt')))
        arm.write_text('own diagnostic\n')
        model.forward(first.input_ids, torch.arange(3), first)
        step = batch([7, 8, 9, 10] if verify else [7], 'TARGET_VERIFY' if verify else 'DECODE')
        model.forward(step.input_ids, torch.arange(3, 3+len(step.input_ids)), step)
        self.assertEqual(recorder.state['status'], 'complete')
        # Later forwards cannot overwrite the two observations.
        model.forward(step.input_ids, torch.arange(3, 3+len(step.input_ids)), step)
        self.assertEqual(len(recorder.state['frames']), 2)
        return recorder

    def test_real_hooks_exclude_unarmed_calls_and_align_last_prefill_first_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.capture(root / 'a')
            b = self.capture(root / 'b', verify=True)
            summary = compare(a.output, b.output)
            self.assertTrue(all(f['shared_dense_bit_identical'] for f in summary['frames']))
            manifest, tensors = read(a.output)
            for event in manifest['frames'][0]['events']:
                for leaf in event['leaves']:
                    self.assertTrue(torch.all(tensors[0][leaf['key']] == 3))
            for event in manifest['frames'][1]['events']:
                for leaf in event['leaves']:
                    self.assertTrue(torch.all(tensors[1][leaf['key']] == 7))

    def test_actual_step_difference_and_incomplete_or_tampered_capture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.capture(root / 'a')
            b = self.capture(root / 'b', offset=1, verify=True)
            summary = compare(a.output, b.output)
            self.assertTrue(summary['frames'][0]['shared_dense_bit_identical'])
            self.assertFalse(summary['frames'][1]['shared_dense_bit_identical'])
            self.assertEqual(summary['frames'][1]['first_difference_among_shared_dense_boundaries']['stage'], 'input')
            path = b.output / 'forward-01.pt'
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'digest'):
                compare(a.output, b.output)
            path = a.output / 'trace.json'
            m = json.loads(path.read_text());m['status'] = 'awaiting_first_step'
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'complete'):
                read(a.output)

    def test_gdn_snapshots_select_actual_slot_and_survive_subsequent_mutations(self):
        cache = SimpleNamespace(conv=[torch.zeros(2, 3, 4)], temporal=torch.zeros(2, 2, 3))
        backend = SimpleNamespace(forward_metadata=SimpleNamespace(mamba_cache_indices=torch.tensor([1])),
                                  state_layer_cache=lambda layer_id: cache)
        context = ModuleType('sglang.srt.model_executor.forward_context')
        context.get_attn_backend = lambda: SimpleNamespace(linear_attn_backend=backend)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {context.__name__: context}):
            recorder = self.capture(Path(tmp), verify=True, cache=cache)
            manifest, tensors = read(recorder.output)
            expected = ['persistent_after_prefill', 'persistent_before_step']
            for i, frame in enumerate(manifest['frames']):
                states = [e for e in frame['events'] if e['stage'].startswith('persistent_')]
                self.assertEqual([e['stage'] for e in states], [expected[i]])
                for leaf in states[0]['leaves']:
                    self.assertTrue(torch.all(tensors[i][leaf['key']] == 3))
            self.assertTrue(torch.all(cache.temporal[1] == 5))
            self.assertTrue(torch.all(cache.temporal[0] == 0))

    def test_quantized_pair_scales_are_opaque_even_when_floating(self):
        for dtype in (torch.float8_e4m3fn, torch.float32):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a = self.capture(root / 'a', opaque_scale=(1, dtype))
                b = self.capture(root / 'b', verify=True, opaque_scale=(2, dtype))
                summary = compare(a.output, b.output)
                self.assertTrue(all(frame['shared_dense_bit_identical'] for frame in summary['frames']))
                self.assertTrue(all(frame['opaque_layouts_not_compared'] for frame in summary['frames']))
                padding = self.capture(root / 'padding', opaque_scale=(float('nan'), dtype))
                self.assertTrue(all(frame['shared_dense_bit_identical']
                                    for frame in compare(a.output, padding.output)['frames']))
                nonfinite = self.capture(root / 'nonfinite', offset=float('nan'))
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    read(nonfinite.output)
                manifest, _ = read(a.output)
                pairs = [event for event in manifest['frames'][0]['events']
                         if event['module'].endswith('mlp.down_proj')]
                self.assertEqual(len(pairs), 2)
                for event in pairs:
                    self.assertEqual(len(event['leaves']), 2)
                    self.assertTrue(all(leaf['comparison_scope'] == 'opaque_layout_requires_validation'
                                        and leaf['selection_scope'] == 'unaligned_physical_slice'
                                        for leaf in event['leaves']))
                # An integer or FP8 capture cannot become a dense comparison
                # merely by relabeling its metadata.
                leaf = pairs[0]['leaves'][0]
                leaf['comparison_scope'] = 'dense_values'
                (a.output / 'trace.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'Quantized layout'):
                    read(a.output)

    def test_budget_exhaustion_is_failed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'byte limit'):
                self.capture(Path(tmp), limit=1)
            manifests = list(Path(tmp).glob('trace/worker-*/trace.json'))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
