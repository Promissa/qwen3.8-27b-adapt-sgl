"""Bounded first-full-attention cache cases through public backbone hooks.

A diagnostic for C1, page-size1 ordinary MHA. It records actual post-write KV
bytes and original metadata; it never reads an entire allocated cache pool.
Used beside forward_trace, preserving that collector and its historical data.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

from common import write_json


class AttentionCases:
    def __init__(self, config):
        self.config = dict(config)
        self.arm = Path(config['arm_file'])
        self.output = Path(config['output_dir']) / f'worker-{os.getpid()}'
        self.limit = int(config.get('max_bytes', 32 * 1024**2))
        self.max_kv_rows = int(config.get('max_kv_rows', 2048))
        if not 0 < self.limit <= 64 * 1024**2 or not 0 < self.max_kv_rows <= 4096:
            raise ValueError('Explicit small attention-case bounds required')
        self.used = 0
        self.installed = False
        self.current = None
        self.values = {}
        self.original_strides = {}
        self.state = {'status': 'unarmed', 'frames': [], 'config': self.config,
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'scope': 'Complete selected-layer Q/K/V/output and logical-order post-write cache bytes for one short C1 prefill and first decode/verify. Synchronizing diagnostic, not a benchmark or losslessness gate.'}

    def save(self):
        write_json(self.output / 'cases.json', self.state)

    def fail(self, reason):
        self.state.update(status='failed', error=reason)
        self.save()
        raise RuntimeError(reason)

    def put(self, name, tensor):
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor) or name in self.values:
            self.fail('Require unique actual tensor fields')
        size = tensor.numel() * tensor.element_size()
        if self.used + size > self.limit:
            self.fail('Attention-case byte limit exceeded')
        self.used += size
        self.original_strides[name] = list(tensor.stride())
        self.values[name] = tensor.detach().to(device='cpu', copy=True).contiguous()

    def before(self, layer, args, kwargs):
        if len(self.state['frames']) >= 2 or not self.arm.exists():
            return
        if self.current is not None:
            self.fail('Reentrant attention case is unsupported')
        batch = kwargs.get('forward_batch', args[3] if len(args) > 3 else None)
        if batch is None or batch.batch_size != 1:
            self.fail('Attention cases require one actual request')
        mode = batch.forward_mode
        phase = 'step' if mode.is_decode() or mode.is_target_verify() else 'prefill'
        if phase != ('prefill' if not self.state['frames'] else 'step'):
            self.fail('Require fresh prefill followed by the first step')
        self.current = {'index': len(self.state['frames']), 'phase': phase,
                        'mode': mode.name, 'layer_id': layer.layer_id,
                        'input_ids': batch.input_ids.detach().cpu().tolist()}
        self.values = {}
        self.original_strides = {}
        self.state['status'] = 'capturing'
        for index, name in enumerate(('q', 'k', 'v')):
            value = kwargs.get(name, args[index] if len(args) > index else None)
            if value is None or value.shape[0] != len(self.current['input_ids']):
                self.fail('Q/K/V must align with all actual input rows')
            self.put(name, value)
        self.put('positions', batch.positions)
        self.put('seq_lens', batch.seq_lens)
        self.put('out_cache_loc', batch.out_cache_loc)
        self.batch = batch

    def after(self, layer, args, output):
        if self.current is None:
            return
        from sglang.srt.model_executor.forward_context import get_attn_backend
        backend = get_attn_backend()
        backend = getattr(backend, 'full_attn_backend', backend)
        if (backend.page_size != 1 or getattr(backend, 'use_mla', False)
                or getattr(backend, 'dcp_size', 1) != 1 or layer.is_cross_attention
                or layer.sliding_window_size > 0 or not backend.enable_deterministic):
            self.fail('Attention cases support page1 C1 full MHA only')
        meta = backend.forward_metadata
        ptr = meta.kv_indptr[:2]
        span = ptr.detach().cpu().tolist()
        if span[0] != 0 or not 0 <= span[1] <= self.max_kv_rows:
            self.fail('Invalid or oversized prefix cache rail')
        prefix = meta.kv_indices[:span[1]]
        out_loc = self.batch.out_cache_loc
        decode = self.batch.forward_mode.is_decode()
        locations = prefix if decode else torch.cat((prefix, out_loc))
        count = locations.numel()
        queries = len(self.current['input_ids'])
        if not 0 < count <= self.max_kv_rows or (decode and queries != 1):
            self.fail('Invalid bounded logical cache span')
        prefix_len = count - queries if decode else prefix.numel()
        if prefix_len < 0:
            self.fail('Decode cache rail omits the current query')
        if decode and not torch.equal(locations[-1:], out_loc):
            self.fail('Decode rail does not end at its actual current KV slot')
        self.put('kv_indptr', ptr)
        self.put('kv_indices', prefix)
        self.put('logical_locations', locations)
        self.put('qo_indptr', meta.qo_indptr)
        self.put('num_kv_splits', meta.num_kv_splits)
        custom = meta.custom_mask
        if custom is not None:
            mask_ptr = meta.mask_indptr[:2]
            bounds = mask_ptr.detach().cpu().tolist()
            if bounds != [0, queries * count]:
                self.fail('Unexpected custom mask span')
            self.put('mask_indptr', mask_ptr)
            self.put('custom_mask', custom[:bounds[1]])
        self.put('output', output)
        cache = backend.kv_layers.read(layer.layer_id)
        cache_meta = {}
        for name, buffer in [('key', cache.key()), ('value', cache.value())]:
            if (buffer.ndim != 3 or buffer.stride(-1) != 1
                    or buffer.dtype not in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32)):
                self.fail('Unsupported cached KV layout')
            # Byte views preserve FP8 values exactly, including their encoding.
            # Index bytes first: CPU/GPU FP8 indexing need not be implemented.
            self.put('cache_' + name + '_bits', buffer.view(torch.uint8).index_select(0, locations.long()))
            cache_meta[name] = {'dtype': str(buffer.dtype), 'shape': list(buffer.shape),
                                'stride': list(buffer.stride())}
        self.current.update(prefix_len=prefix_len, cache=cache_meta,
            q_heads=layer.tp_q_head_num, kv_heads=layer.tp_k_head_num,
            qk_head_dim=layer.qk_head_dim, v_head_dim=layer.v_head_dim,
            scaling=layer.scaling,
            k_scale=layer.k_scale_float if layer.k_scale is not None and layer.v_scale is not None else 1.0,
            v_scale=layer.v_scale_float if layer.k_scale is not None and layer.v_scale is not None else 1.0,
            logit_cap=layer.logit_cap, xai_temperature_len=layer.xai_temperature_len,
            max_kv_splits=backend.max_kv_splits,
            split_tile_size=backend.split_tile_size,
            enable_deterministic=backend.enable_deterministic,
            use_pdl=getattr(backend, 'use_pdl', False),
            original_tensor_strides=self.original_strides,
            tensor_metadata={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in self.values.items()})
        filename = f'case-{self.current["index"]:02d}.pt'
        path = self.output / filename
        torch.save(self.values, path)
        self.current.update(tensor_file=filename, tensor_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.state['frames'].append(self.current)
        self.state.update(status='complete' if len(self.state['frames']) == 2 else 'awaiting_first_step',
                          captured_tensor_bytes=self.used)
        self.current = None
        self.values = {}
        self.batch = None
        self.save()

    def bootstrap(self, module, args, output):
        if self.installed:
            return
        self.installed = True
        if type(module).__name__ != self.config.get('expected_model_class', 'Qwen3_5ForCausalLM'):
            return
        self.output.mkdir(parents=True, exist_ok=False)
        name = f'layers.{int(self.config.get("layer_id", 3))}.attn'
        layer = dict(module.named_modules()).get(name)
        if layer is None or layer.layer_id != int(self.config.get('layer_id', 3)):
            self.fail('Requested full-attention layer was not found')
        layer.register_forward_pre_hook(self.before, with_kwargs=True)
        layer.register_forward_hook(self.after)
        self.state['module'] = name
        self.save()


def make_hook(config):
    return AttentionCases(config).bootstrap


def read_cases(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'cases.json').read_text())
    if (meta['status'] != 'complete' or [f['phase'] for f in meta['frames']] != ['prefill', 'step']
            or meta['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Require complete cases from the current collector source')
    values = []
    total_bytes = 0
    for frame in meta['frames']:
        path = directory / frame['tensor_file']
        if path.resolve().parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != frame['tensor_sha256']:
            raise ValueError('Attention-case artifact path/digest mismatch')
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        if set(tensors) != set(frame['tensor_metadata']):
            raise ValueError('Attention-case tensor coverage mismatch')
        for name, value in tensors.items():
            if {'shape': list(value.shape), 'dtype': str(value.dtype)} != frame['tensor_metadata'][name]:
                raise ValueError('Attention-case tensor metadata mismatch')
            if value.is_floating_point() and not torch.isfinite(value.float()).all():
                raise ValueError('Nonfinite attention input/output')
            total_bytes += value.numel() * value.element_size()
        visibility(frame, tensors)
        values.append(tensors)
    if total_bytes != meta['captured_tensor_bytes'] or total_bytes > meta['config'].get('max_bytes', 32 * 1024**2):
        raise ValueError('Attention-case byte coverage mismatch')
    return meta, values


def visibility(frame, tensors):
    rows = len(frame['input_ids'])
    count = len(tensors['logical_locations'])
    prefix = frame['prefix_len']
    if prefix < 0 or count != prefix + rows:
        raise ValueError('Logical cache does not cover prefix and current rows')
    if 'custom_mask' in tensors:
        mask = tensors['custom_mask'].bool().reshape(rows, count)
    else:
        mask = torch.arange(count)[None, :] <= prefix + torch.arange(rows)[:, None]
    return mask


def compare_cases(reference, candidate):
    a, av = read_cases(reference)
    b, bv = read_cases(candidate)
    if a['frames'][0]['input_ids'] != b['frames'][0]['input_ids']:
        raise ValueError('Different actual attention prompt IDs')
    reports = []
    for af, bf, at, bt in zip(a['frames'], b['frames'], av, bv):
        if (af['prefix_len'] != bf['prefix_len'] or af['input_ids'][0] != bf['input_ids'][0]
                or at['positions'][0] != bt['positions'][0]):
            raise ValueError('Unaligned actual attention inputs')
        if af['phase'] == 'prefill' and not torch.equal(at['positions'], bt['positions']):
            raise ValueError('Different actual prefill positions')
        if any(af['cache'][k]['dtype'] != bf['cache'][k]['dtype'] for k in ('key', 'value')):
            raise ValueError('Different stored KV dtypes')
        for key in ['q_heads', 'kv_heads', 'qk_head_dim', 'v_head_dim', 'scaling', 'k_scale', 'v_scale']:
            if af[key] != bf[key]:
                raise ValueError('Different attention geometry or cache scales')
        am, bm = visibility(af, at)[0], visibility(bf, bt)[0]
        def equal(x, y):
            return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        reports.append({'phase': af['phase'], 'prefix_len': af['prefix_len'],
            'all_prefill_cache_key_bits_equal': equal(at['cache_key_bits'], bt['cache_key_bits']) if af['phase'] == 'prefill' else None,
            'all_prefill_cache_value_bits_equal': equal(at['cache_value_bits'], bt['cache_value_bits']) if af['phase'] == 'prefill' else None,
            'first_query_bits_equal': equal(at['q'][:1], bt['q'][:1]),
            'first_output_bits_equal': equal(at['output'][:1], bt['output'][:1]),
            'first_query_visible_key_bits_equal': equal(at['cache_key_bits'][am], bt['cache_key_bits'][bm]),
            'first_query_visible_value_bits_equal': equal(at['cache_value_bits'][am], bt['cache_value_bits'][bm]),
            'first_query_visible_tokens': int(am.sum()),
            'candidate_first_query_visible_tokens': int(bm.sum()),
            'reference_first_query_matches_causal_visibility': bool(torch.equal(am, torch.arange(len(am)) <= af['prefix_len'])),
            'candidate_first_query_matches_causal_visibility': bool(torch.equal(bm, torch.arange(len(bm)) <= bf['prefix_len']))})
    return {'reference': str(reference), 'candidate': str(candidate), 'frames': reports,
            'scope': 'Actual selected-layer cached bytes and masks; no causal kernel replay, timing or full-model losslessness claim.'}


def compare_pair(output):
    """Bind cached cases to the already validated real-prompt capture."""
    output = Path(output)
    directories = []
    for role in ('control', 'ngram'):
        workers = list((output / role / 'attention-cases').glob('worker-*'))
        if len(workers) != 1:
            raise ValueError('Require exactly one attention-case target worker per role')
        meta, values = read_cases(workers[0])
        capture = json.loads((output / role / 'capture.json').read_text())
        prompt_ids = capture['prompt_tokenization']['tokens']
        if (capture['status'] != 'passed' or meta['frames'][0]['input_ids'] != prompt_ids
                or values[0]['positions'].tolist() != list(range(len(prompt_ids)))
                or meta['frames'][1]['input_ids'][0] != capture['output_ids'][0]
                or values[1]['positions'][0].item() != len(prompt_ids)):
            raise ValueError('Attention cache cases do not belong to the validated prompt')
        directories.append(workers[0])
    return compare_cases(*directories)
