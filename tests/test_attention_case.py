"""Actual hooks capture logical cached FP8 bytes, not arbitrary pool rows."""
import ast
import copy
import hashlib
import importlib.util
import json
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
from attention_case import read_cases, compare_cases

spec = importlib.util.spec_from_file_location('case_actual_hooks', ROOT /
    'engine/sglang/python/sglang/srt/model_executor/hook_manager.py')
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == 'DECODE'

    def is_target_verify(self):
        return self.name == 'TARGET_VERIFY'


class Attention(nn.Module):
    layer_id = 3
    is_cross_attention = False
    sliding_window_size = -1
    tp_q_head_num = 2
    tp_k_head_num = 1
    qk_head_dim = v_head_dim = 4
    scaling = .5
    k_scale = v_scale = None
    k_scale_float = v_scale_float = None
    logit_cap = 0.
    xai_temperature_len = -1

    def __init__(self, backend, cache):
        super().__init__()
        self.backend, self.cache = backend, cache

    def forward(self, q, k, v, forward_batch):
        b = forward_batch
        self.backend.forward_metadata = b.metadata
        # CPU float8 indexing is unsupported on some builds; storing its exact
        # uint8 view exercises the same physical cache encoding independently.
        for tensor, value in [(self.cache.key(), k), (self.cache.value(), v)]:
            tensor.view(torch.uint8)[b.out_cache_loc] = value.reshape(-1, 1, 4).to(tensor.dtype).view(torch.uint8)
        return q * .125


class Backbone(nn.Module):
    def __init__(self, backend, cache):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        self.layers[3].attn = Attention(backend, cache)

    def forward(self, input_ids, positions, forward_batch):
        backing = torch.zeros(len(input_ids), 12, dtype=torch.bfloat16)
        backing[:, :8] = input_ids.to(torch.bfloat16)[:, None]
        q = backing[:, :8]
        k = input_ids.to(torch.bfloat16)[:, None].expand(-1, 4).contiguous()
        return self.layers[3].attn(q, k, k + 1, forward_batch)


class Outer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone

    def forward(self, b):
        return self.model(input_ids=b.input_ids, positions=b.positions, forward_batch=b)


def batch(mode, ids, locations, prefix):
    step = mode != 'EXTEND'
    n = len(ids)
    locations = torch.tensor(locations, dtype=torch.int64)
    kv = torch.tensor(prefix, dtype=torch.int64)
    custom = None
    if mode == 'TARGET_VERIFY':
        custom = torch.cat((torch.ones(n, len(prefix), dtype=torch.bool),
                            torch.ones(n, n, dtype=torch.bool).tril()), dim=1).flatten()
    metadata = SimpleNamespace(kv_indptr=torch.tensor([0, len(kv)], dtype=torch.int32), kv_indices=kv,
        qo_indptr=None if mode == 'DECODE' else torch.tensor([0, n], dtype=torch.int32),
        num_kv_splits=torch.tensor([1], dtype=torch.int32) if mode == 'DECODE' else None,
        custom_mask=custom, mask_indptr=torch.tensor([0, n * (len(prefix) + n)], dtype=torch.int32))
    return SimpleNamespace(input_ids=torch.tensor(ids), positions=torch.arange(3 if step else 0, (3 if step else 0) + n),
        seq_lens=torch.tensor([4 if mode == 'DECODE' else 3 if step else n], dtype=torch.int32),
        out_cache_loc=locations, forward_mode=Mode(mode), batch_size=1, metadata=metadata)


class AttentionCaseTests(unittest.TestCase):
    def capture(self, root, verify=False, limit=32 * 1024**2):
        key = torch.zeros(20, 1, 4, dtype=torch.float8_e4m3fn)
        value = torch.zeros_like(key)
        cache = SimpleNamespace(key=lambda: key, value=lambda: value)
        backend = SimpleNamespace(page_size=1, dcp_size=1, use_mla=False,
            max_kv_splits=8, split_tile_size=256, enable_deterministic=True,
            kv_layers=SimpleNamespace(read=lambda layer_id: cache))
        context = ModuleType('sglang.srt.model_executor.forward_context')
        context.get_attn_backend = lambda: SimpleNamespace(full_attn_backend=backend)
        model = Outer(Backbone(backend, cache))
        arm = root / 'armed'
        config = {'output_dir': str(root / 'cases'), 'arm_file': str(arm),
                  'expected_model_class': 'Backbone', 'max_bytes': limit}
        hooks.register_forward_hooks(model, [{'target_modules': ['model'],
            'hook_factory': 'attention_case:make_hook', 'config': config}])
        recorder = next(iter(model.model._forward_hooks.values())).__self__
        prefill = batch('EXTEND', [1, 2, 3], [2, 5, 7], [])
        with patch.dict(sys.modules, {context.__name__: context}):
            model.forward(prefill)
            model.forward(prefill)
            self.assertEqual(recorder.state['frames'], [])
            arm.write_text('owned prompt\n')
            model.forward(prefill)
            step = batch('TARGET_VERIFY', [4, 5, 6, 7], [11, 13, 14, 15], [2, 5, 7]) if verify else batch('DECODE', [4], [11], [2, 5, 7, 11])
            model.forward(step)
            model.forward(step)
        return recorder, cache

    def test_actual_hooks_record_complete_logical_bytes_and_survive_cache_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, cache = self.capture(root / 'control')
            b, _ = self.capture(root / 'verify', verify=True)
            report = compare_cases(a.output, b.output)
            self.assertTrue(all(f['first_query_visible_key_bits_equal'] and f['first_query_visible_value_bits_equal'] for f in report['frames']))
            self.assertTrue(all(f['candidate_first_query_matches_causal_visibility'] for f in report['frames']))
            manifest, values = read_cases(a.output)
            self.assertEqual(values[0]['logical_locations'].tolist(), [2, 5, 7])
            self.assertEqual(values[1]['logical_locations'].tolist(), [2, 5, 7, 11])
            self.assertEqual(values[1]['cache_key_bits'].view(torch.float8_e4m3fn).float()[:, 0, 0].tolist(), [1, 2, 3, 4])
            cache.key().view(torch.uint8).fill_(255)
            self.assertEqual(read_cases(a.output)[1][1]['cache_key_bits'].view(torch.float8_e4m3fn).float()[:, 0, 0].tolist(), [1, 2, 3, 4])
            self.assertEqual(len(manifest['frames']), 2)

    def test_actual_mask_difference_is_reported_and_corrupt_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, _ = self.capture(root / 'control')
            b, _ = self.capture(root / 'verify', verify=True)
            meta, values = read_cases(b.output)
            values[1]['custom_mask'][4] = True  # root sees an extra future node
            path = b.output / meta['frames'][1]['tensor_file']
            torch.save(values[1], path)
            meta['frames'][1]['tensor_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (b.output / 'cases.json').write_text(json.dumps(meta))
            report = compare_cases(a.output, b.output)
            self.assertFalse(report['frames'][1]['candidate_first_query_matches_causal_visibility'])
            self.assertFalse(report['frames'][1]['first_query_visible_key_bits_equal'])
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'digest'):
                read_cases(b.output)

    def test_extra_case_plan_is_explicit_and_keeps_the_default_trace_hook(self):
        from trace_pair import plans
        with tempfile.TemporaryDirectory() as tmp:
            ordinary = plans(Path(tmp))
            extended = plans(Path(tmp), attention_cases=True)
            for old, new in zip(ordinary, extended):
                oh = json.loads(old['command'][old['command'].index('--forward-hooks') + 1])
                nh = json.loads(new['command'][new['command'].index('--forward-hooks') + 1])
                self.assertEqual(len(oh), 1)
                self.assertEqual(nh[0], oh[0])
                self.assertEqual(len(nh), 2)
                self.assertEqual(nh[1]['hook_factory'], 'attention_case:make_hook')
                self.assertEqual(nh[1]['config']['arm_file'], oh[0]['config']['arm_file'])
                self.assertTrue(nh[1]['config']['output_dir'].endswith('/attention-cases'))

    def test_queued_boundary_refuses_a_live_model_and_requires_a_free_port(self):
        source = ROOT / 'results/driver-sources/numerical-backend-followups-20261111.py'
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_gpu_boundary')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            proc = folder / 'proc/101'
            proc.mkdir(parents=True)
            (proc / 'cmdline').write_bytes(b'python\0-m\0sglang.launch_server\0')
            binds = []
            busy = False

            class Socket:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def setsockopt(self, *args):
                    pass

                def bind(self, address):
                    binds.append(address)
                    if busy:
                        raise OSError('port occupied')

            namespace = {'Path': lambda value: folder / 'proc' if value == '/proc' else Path(value),
                         'socket': SimpleNamespace(socket=Socket, SOL_SOCKET=1, SO_REUSEADDR=2)}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
            with self.assertRaisesRegex(RuntimeError, 'Model remains active'):
                namespace['check_gpu_boundary']()
            self.assertEqual(binds, [])
            (proc / 'cmdline').write_bytes(b'python\0read-only-check.py\0')
            namespace['check_gpu_boundary']()
            self.assertEqual(binds, [('127.0.0.1', 30000)])
            busy = True
            with self.assertRaisesRegex(OSError, 'port occupied'):
                namespace['check_gpu_boundary']()

    def test_case_budget_failure_withholds_complete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'byte limit'):
                self.capture(Path(tmp), limit=1)
            manifest, = Path(tmp).glob('cases/worker-*/cases.json')
            self.assertEqual(json.loads(manifest.read_text())['status'], 'failed')
            with self.assertRaisesRegex(ValueError, 'complete cases'):
                read_cases(manifest.parent)

    def test_replay_rebuild_preserves_actual_strides_and_physical_fp8_slots(self):
        from replay_attention_case import rebuild
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.capture(Path(tmp), verify=True)
            meta, values = read_cases(recorder.output)
            frame, tensors = meta['frames'][1], values[1]
            rebuilt = rebuild(frame, tensors)
            self.assertEqual(rebuilt['q'].stride(), (12, 1))
            self.assertTrue(torch.equal(rebuilt['q'], tensors['q']))
            for name in ('key', 'value'):
                self.assertEqual(list(rebuilt[name].stride()), frame['cache'][name]['stride'])
                raw = rebuilt[name].view(torch.uint8)
                self.assertTrue(torch.equal(raw[tensors['logical_locations']], tensors['cache_' + name + '_bits']))
                missing = sorted(set(range(len(raw))) - set(tensors['logical_locations'].tolist()))
                self.assertEqual(int(raw[missing].count_nonzero()), 0)
                raw.zero_()
                self.assertNotEqual(int(tensors['cache_' + name + '_bits'].count_nonzero()), 0)
            self.assertEqual(int(rebuilt['output'].count_nonzero()), 0)

    def test_replay_uses_actual_branch_visibility_and_rejects_unrepresentable_positions(self):
        from replay_attention_case import decode_metadata
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.capture(Path(tmp), verify=True)
            meta, values = read_cases(recorder.output)
            frame, tensors = meta['frames'][1], values[1]
            mask = tensors['custom_mask'].reshape(4, 7)
            mask[:, 3:] = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]])
            tensors['positions'] = torch.tensor([3, 4, 4, 5])
            rails = decode_metadata(frame, tensors)
            self.assertEqual(rails['kv_indptr'].tolist(), [0, 4, 9, 14, 20])
            self.assertEqual(rails['kv_indices'][9:14].tolist(), [2, 5, 7, 11, 14])
            self.assertEqual(rails['kv_indices'][14:20].tolist(), [2, 5, 7, 11, 14, 15])
            frame['split_tile_size'] = 2
            self.assertEqual(decode_metadata(frame, tensors)['num_kv_splits'].tolist(), [2, 3, 3, 3])
            frame['max_kv_splits'] = 2
            with self.assertRaisesRegex(ValueError, 'split count'):
                decode_metadata(frame, tensors)
            frame['max_kv_splits'] = 8
            frame['split_tile_size'] = 0
            with self.assertRaisesRegex(ValueError, 'split tile'):
                decode_metadata(frame, tensors)
            frame['split_tile_size'] = 256
            tensors['positions'][2] = 5
            with self.assertRaisesRegex(ValueError, 'mask/positions'):
                decode_metadata(frame, tensors)
            tensors['positions'][2] = 4
            mask[0, 0] = False
            with self.assertRaisesRegex(ValueError, 'mask/positions'):
                decode_metadata(frame, tensors)

    def test_replay_rejects_pool_overflow_bad_slots_and_conflicting_duplicate_bytes(self):
        from replay_attention_case import rebuild
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.capture(Path(tmp))
            meta, values = read_cases(recorder.output)
            frame, tensors = meta['frames'][1], values[1]
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                rebuild(frame, tensors, max_bytes=1)
            changed = copy.deepcopy(frame)
            changed['original_tensor_strides']['q'] = [1, 1]
            # The decode frame has one row: use an overlapping feature stride.
            changed['original_tensor_strides']['q'][-1] = 0
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                rebuild(changed, tensors)
            changed = copy.deepcopy(tensors)
            changed['logical_locations'][0] = frame['cache']['key']['shape'][0]
            with self.assertRaisesRegex(ValueError, 'physical locations'):
                rebuild(frame, changed)
            changed['logical_locations'][0] = changed['logical_locations'][1]
            with self.assertRaisesRegex(ValueError, 'Contradictory'):
                rebuild(frame, changed)

    def test_decode_replay_preserves_output_layout_and_owns_each_query_result(self):
        from replay_attention_case import decode_metadata, rebuild, run_decode
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.capture(Path(tmp), verify=True)
            meta, values = read_cases(recorder.output)
            frame, tensors = meta['frames'][1], values[1]
            frame['original_tensor_strides']['output'] = [12, 1]
            buffers = rebuild(frame, tensors)
            rails = decode_metadata(frame, tensors)
            kernel = ModuleType('sglang.kernels.ops.attention.decode_attention')
            strides = []

            def fake_decode(q, key, value, out, *args, **kwargs):
                strides.append(out.stride())
                out.copy_(q)

            kernel.decode_attention_fwd = fake_decode
            with patch.dict(sys.modules, {kernel.__name__: kernel}):
                batched = run_decode(frame, tensors, buffers, rails)
                outputs = []
                for index in range(4):
                    start, end = rails['kv_indptr'][index:index+2].tolist()
                    rail = {'kv_indptr': torch.tensor([0, end-start], dtype=torch.int32),
                            'kv_indices': rails['kv_indices'][start:end],
                            'num_kv_splits': rails['num_kv_splits'][index:index+1]}
                    outputs.append(run_decode(frame, tensors, buffers, rail, buffers['q'][index:index+1]))
            self.assertTrue(torch.equal(batched, torch.cat(outputs)))
            # Serving also uses view(), which normalizes the singleton row
            # stride; the batched output must retain its padded row layout.
            self.assertEqual(strides, [(12, 4, 1)] + [(8, 4, 1)] * 4)
            outputs[-1].zero_()
            self.assertEqual(outputs[0][0, 0].item(), 4)

    def test_unified_tile_experiment_changes_only_kv_tile_and_restores_dispatch(self):
        from replay_attention_case import rebuild, run_unified, run_unified_tile
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.capture(Path(tmp), verify=True)
            meta, values = read_cases(recorder.output)
            frame, tensors = meta['frames'][1], values[1]
            buffers = rebuild(frame, tensors)
            native = ModuleType('sglang.kernels.ops.attention.extend_attention')
            original = (4, 0, 4, 64, 64, 4)
            getter = lambda q_dim, v_dim: original
            native._get_block_sizes_for_extend_attention = getter
            observed = []
            mask = tensors['custom_mask'].clone()

            def launch(q, out, key, value, *args, **kwargs):
                config = native._get_block_sizes_for_extend_attention(q.shape[-1], value.shape[-1])
                observed.append(config)
                self.assertTrue(torch.equal(kwargs['custom_mask'], mask))
                out.copy_(q * .125)

            native.extend_attention_fwd_unified = launch
            with patch.dict(sys.modules, {native.__name__: native}):
                output, configuration = run_unified_tile(frame, tensors, buffers, 32)
                self.assertTrue(torch.equal(output, tensors['output']))
                self.assertEqual(configuration['selected']['BLOCK_N'], 32)
                self.assertEqual(configuration['selector_calls'], 1)
                self.assertIs(native._get_block_sizes_for_extend_attention, getter)
                run_unified(frame, tensors, buffers)
                self.assertEqual(observed, [(4, 0, 4, 64, 32, 4), original])
                self.assertTrue(torch.equal(tensors['custom_mask'], mask))

                def fail(q, out, key, value, *args, **kwargs):
                    native._get_block_sizes_for_extend_attention(q.shape[-1], value.shape[-1])
                    raise RuntimeError('launch failed')

                native.extend_attention_fwd_unified = fail
                with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                    run_unified_tile(frame, tensors, buffers, 32)
                self.assertIs(native._get_block_sizes_for_extend_attention, getter)
                with self.assertRaisesRegex(ValueError, 'bounded'):
                    run_unified_tile(frame, tensors, buffers, 16)


if __name__ == '__main__':
    unittest.main()
