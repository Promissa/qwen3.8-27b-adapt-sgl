#!/usr/bin/env python3
"""Replay actual cached-KV cases only at a clean GPU boundary.

CPU reconstruction/preflight is not GPU parity. Kernel-route conclusions require
both original kernels to reproduce their saved outputs, then identical-input
cross replay. This contains no engine change, timing benchmark or promotion.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

import torch

from attention_case import compare_pair, read_cases, visibility
from common import ROOT, now, write_json
from trace_pair import retained_sources

CACHE_DTYPES = {str(d): d for d in (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32)}
MAX_REBUILD_BYTES = 256 * 1024**2
MAX_SCRATCH_BYTES = 256 * 1024**2


def storage_bytes(shape, stride, element_size):
    if (len(shape) != len(stride) or not shape or any(type(x) is not int or x <= 0 for x in shape)
            or any(type(x) is not int or x < 0 for x in stride)):
        raise ValueError('Invalid recorded shape/stride')
    span = 1
    for step, size in sorted((s, n) for n, s in zip(shape, stride) if n > 1):
        if step < span:
            raise ValueError('Unsupported overlapping or interleaved recorded strides')
        span += (size - 1) * step
    return span * element_size


def rebuild(frame, tensors, device='cpu', max_bytes=MAX_REBUILD_BYTES):
    rows = len(frame['input_ids'])
    if frame['q_heads'] <= 0 or frame['kv_heads'] <= 0 or frame['q_heads'] % frame['kv_heads']:
        raise ValueError('Invalid recorded head grouping')
    if (tensors['q'].numel() != rows * frame['q_heads'] * frame['qk_head_dim']
            or tensors['output'].numel() != rows * frame['q_heads'] * frame['v_head_dim']):
        raise ValueError('Recorded Q/output geometry differs')
    locations = tensors['logical_locations'].long()
    result, budget, layouts = {}, 0, {}
    for name in ('q', 'output'):
        value = tensors[name]
        stride = frame['original_tensor_strides'][name]
        shape = list(value.shape)
        budget += storage_bytes(shape, stride, value.element_size())
        layouts[name] = (shape, stride, value.dtype)
    for name, dim in [('key', frame['qk_head_dim']), ('value', frame['v_head_dim'])]:
        item = frame['cache'][name]
        if item['dtype'] not in CACHE_DTYPES:
            raise ValueError('Unsupported cached dtype')
        dtype = CACHE_DTYPES[item['dtype']]
        size = torch.empty((), dtype=dtype).element_size()
        shape, stride = item['shape'], item['stride']
        if (len(shape) != 3 or shape[1:] != [frame['kv_heads'], dim] or stride[-1] != 1
                or locations.ndim != 1 or not len(locations)
                or locations.min().item() < 0 or locations.max().item() >= shape[0]
                or tensors['cache_' + name + '_bits'].dtype != torch.uint8
                or list(tensors['cache_' + name + '_bits'].shape) != [len(locations), shape[1], dim * size]):
            raise ValueError('Invalid cache layout, byte shape or physical locations')
        budget += storage_bytes(shape, stride, size)
        layouts[name] = (shape, stride, dtype)
        # Repeated physical slots are legitimate only when their saved bytes
        # agree. Never resolve contradictory writes by an arbitrary last value.
        first = {}
        for index, slot in enumerate(locations.tolist()):
            if slot in first and not torch.equal(tensors['cache_' + name + '_bits'][index],
                                                 tensors['cache_' + name + '_bits'][first[slot]]):
                raise ValueError('Contradictory captured bytes for one physical slot')
            first[slot] = index
    if budget > max_bytes:
        raise ValueError('Recorded pool reconstruction exceeds the bounded byte budget')
    for name, (shape, stride, dtype) in layouts.items():
        value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        if name in ('q', 'output'):
            if name == 'q':
                value.copy_(tensors[name].to(device))
            else:
                value.zero_()
        else:
            value.view(torch.uint8).zero_()
            value.view(torch.uint8).index_copy_(0, locations.to(device), tensors['cache_' + name + '_bits'].to(device))
        result[name] = value
    result['allocated_storage_bytes'] = budget
    return result


def decode_metadata(frame, tensors):
    """One decode-style KV rail per query, using the observed visibility mask."""
    mask = visibility(frame, tensors)
    prefix = frame['prefix_len']
    counts = mask.sum(1).to(torch.int32)
    if (not mask[:, :prefix].all() or not torch.equal(tensors['positions'].long(), counts.long() - 1)):
        raise ValueError('Actual mask/positions cannot be represented as ordinary full-prefix decode')
    tile = frame['split_tile_size']
    if not frame['enable_deterministic'] or type(tile) is not int or tile <= 0:
        raise ValueError('Decode replay requires the recorded deterministic split tile')
    splits = (counts + tile - 1) // tile
    if splits.min().item() <= 0 or splits.max().item() > frame['max_kv_splits']:
        raise ValueError('Derived decode split count exceeds recorded storage')
    locations = tensors['logical_locations']
    indices = torch.cat([locations[row] for row in mask])
    ptr = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)])
    return {'kv_indptr': ptr, 'kv_indices': indices, 'num_kv_splits': splits}


def stats(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = bool(torch.isfinite(actual.float()).all())
    equal = compatible and torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    result = {'compatible': compatible, 'finite': finite, 'bit_identical': bool(equal)}
    if compatible and finite:
        delta = (actual.float() - expected.float()).abs()
        result.update(different_values=int((actual.float() != expected.float()).sum()),
                      max_abs_difference=float(delta.max()) if delta.numel() else 0., numel=actual.numel())
    return result


def run_decode(frame, tensors, buffers, metadata, query=None):
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    q = (buffers['q'] if query is None else query).view(-1, frame['q_heads'], frame['qk_head_dim'])
    rows = q.shape[0]
    splits = frame['max_kv_splits']
    scratch = rows * frame['q_heads'] * splits * (frame['v_head_dim'] + 1) * 4
    if scratch > MAX_SCRATCH_BYTES:
        raise ValueError('Decode scratch exceeds bounded replay budget')
    # Rebuild the observed output layout, then use the serving backend's view
    # operation (which can normalize singleton strides). Each invocation owns
    # its output so later queries cannot
    # overwrite earlier results before the batch/single comparison.
    shape = list(tensors['output'].shape)
    shape[0] = rows
    stride = frame['original_tensor_strides']['output']
    if storage_bytes(shape, stride, tensors['output'].element_size()) > MAX_REBUILD_BYTES:
        raise ValueError('Decode output exceeds bounded replay budget')
    out = torch.empty_strided(shape, stride, dtype=tensors['output'].dtype, device=q.device).view(
        rows, frame['q_heads'], frame['v_head_dim'])
    mid = torch.empty((rows, frame['q_heads'], splits, frame['v_head_dim']), dtype=torch.float32, device=q.device)
    lse = torch.empty((rows, frame['q_heads'], splits), dtype=torch.float32, device=q.device)
    decode_attention_fwd(q, buffers['key'], buffers['value'], out,
        metadata['kv_indptr'].to(q.device), metadata['kv_indices'].to(q.device), mid, lse,
        metadata['num_kv_splits'].to(q.device), splits, frame['scaling'], frame['k_scale'], frame['v_scale'],
        logit_cap=frame['logit_cap'], xai_temperature_len=frame['xai_temperature_len'],
        has_mla=False, use_pdl=frame['use_pdl'], page_size=1, enable_lean=False,
        uneven_batch=torch.zeros(1, dtype=torch.int32, device=q.device))
    return out.reshape(rows, -1)


def run_unified(frame, tensors, buffers):
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd_unified
    q = buffers['q'].view(-1, frame['q_heads'], frame['qk_head_dim'])
    out = buffers['output'].view(-1, frame['q_heads'], frame['v_head_dim'])
    device = q.device
    rows = q.shape[0]
    ptr = torch.tensor([0, len(tensors['logical_locations'])], dtype=torch.int32, device=device)
    extend_attention_fwd_unified(q, out, buffers['key'], buffers['value'], frame['k_scale'], frame['v_scale'],
        torch.tensor([0, rows], dtype=torch.int32, device=device), ptr,
        tensors['logical_locations'].to(device), torch.tensor([frame['prefix_len']], dtype=torch.int32, device=device),
        rows, custom_mask=tensors['custom_mask'].to(device) if 'custom_mask' in tensors else None,
        mask_indptr=tensors['mask_indptr'].to(device) if 'mask_indptr' in tensors else None,
        sm_scale=frame['scaling'], logit_cap=frame['logit_cap'], is_causal=True,
        sliding_window_size=-1, xai_temperature_len=frame['xai_temperature_len'], page_size=1)
    return out.reshape(rows, -1)


def run_unified_tile(frame, tensors, buffers, block_n):
    """Change only the KV tile for an isolated native unified-kernel replay."""
    if type(block_n) is not int or block_n not in (32, 64):
        raise ValueError('Only bounded32/64 KV tile experiments are supported')
    module = importlib.import_module('sglang.kernels.ops.attention.extend_attention')
    dimensions = (frame['qk_head_dim'], frame['v_head_dim'])
    original = tuple(module._get_block_sizes_for_extend_attention(*dimensions))
    if len(original) != 6:
        raise ValueError('Native unified configuration contract changed')
    selected = (*original[:4], block_n, original[5])
    calls = 0

    def configuration(q_dim, v_dim):
        nonlocal calls
        if (q_dim, v_dim) != dimensions:
            raise ValueError('Unexpected geometry during isolated tile replay')
        calls += 1
        return selected

    # The idle-boundary process owns this experiment. The native dispatcher
    # reads this selector once synchronously at launch. Restore it even when
    # compilation/launch fails; the serving checkout is never edited.
    with patch.object(module, '_get_block_sizes_for_extend_attention', configuration):
        output = run_unified(frame, tensors, buffers)
    if calls != 1:
        raise ValueError('Tile experiment did not use the native configuration selector exactly once')
    names = ('BLOCK_DMODEL', 'BLOCK_DPE', 'BLOCK_DV', 'BLOCK_M', 'BLOCK_N', 'num_warps')
    return output, {'original': dict(zip(names, original)), 'selected': dict(zip(names, selected)),
                    'selector_calls': calls,
                    'scope': 'Only BLOCK_N is overridden for this isolated replay; no serving-source edit or timing claim'}


def replay_process(args):
    """Match a Python replay process, not a timeout supervisor's arguments."""
    if not args or not Path(os.fsdecode(args[0])).name.startswith('python'):
        return False
    index = 2 if len(args) > 1 and args[1] == b'-u' else 1
    return len(args) > index and args[index].endswith(b'/replay_attention_case.py')


def check_idle():
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            args = item.joinpath('cmdline').read_bytes().split(b'\0')
            if (b'sglang.launch_server' in args or args[0].startswith(b'sglang::')
                    or replay_process(args)):
                raise RuntimeError('GPU model/replay process remains active; wait for its owner')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 30000))


def load_pair(source):
    pair = json.loads((source / 'pair.json').read_text())
    if pair['status'] != 'completed_diagnostic' or pair.get('attention_cases') is not True:
        raise ValueError('Require completed actual attention-cache capture')
    provenance = retained_sources(pair)
    report = compare_pair(source)
    cases, evidence = {}, {}
    for role in ('control', 'ngram'):
        probe = json.loads((source / role / 'probe/probe.json').read_text())
        engine = probe['runtime_provenance']
        revision = subprocess.check_output(['git', '-C', engine['engine_source'], 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', '-C', engine['engine_source'], 'diff', '--binary', 'HEAD'])
        if revision != engine['engine_commit'] or hashlib.sha256(diff).hexdigest() != engine['engine_diff_sha256']:
            raise ValueError('Replay engine source differs from captured serving source')
        directory, = (source / role / 'attention-cases').glob('worker-*')
        cases[role] = read_cases(directory)
        evidence[role] = {'case_manifest_sha256': hashlib.sha256((directory / 'cases.json').read_bytes()).hexdigest(),
                          'tensor_sha256': [f['tensor_sha256'] for f in cases[role][0]['frames']],
                          'engine_commit': revision, 'engine_diff_sha256': hashlib.sha256(diff).hexdigest()}
    return cases, {'source_manifest_sha256': hashlib.sha256((source / 'pair.json').read_bytes()).hexdigest(),
                   'runner_sources': provenance, 'case_evidence': evidence, 'actual_cache_comparison': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing replay evidence; use a fresh output')
    cases, provenance = load_pair(args.input.resolve())
    result = {'at': now(), 'status': 'running', 'gpu_replay_executed': False, **provenance,
        'replay_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Selected real attention cases only; original-output reproduction precedes any route attribution. No timing, engine patch or full-model losslessness promotion.'}
    write_json(args.output, result)
    try:
        if args.cpu_preflight:
            for meta, values in cases.values():
                for frame, tensors in zip(meta['frames'], values):
                    rebuild(frame, tensors, device='cpu')
            result.update(status='cpu_preflight_passed', gpu_replay_executed=False)
        else:
            check_idle()
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable at the clean replay boundary')
            result['gpu_replay_executed'] = True
            originals, original_outputs = {}, {}
            for role, (meta, values) in cases.items():
                originals[role] = []
                original_outputs[role] = []
                for frame, tensors in zip(meta['frames'], values):
                    buffers = rebuild(frame, tensors, device='cuda')
                    if frame['mode'] == 'DECODE':
                        metadata = {k: tensors[k] for k in ('kv_indptr', 'kv_indices', 'num_kv_splits')}
                        output = run_decode(frame, tensors, buffers, metadata)
                    else:
                        output = run_unified(frame, tensors, buffers)
                    torch.cuda.synchronize()
                    saved = tensors['output'].reshape(len(frame['input_ids']), -1)
                    originals[role].append(stats(output, saved))
                    original_outputs[role].append(output.detach().cpu())
                    del buffers, output
            result['original_kernel_reproduction'] = originals
            reproduced = all(row['bit_identical'] for rows in originals.values() for row in rows)
            result['all_original_outputs_reproduced'] = reproduced
            if reproduced:
                frame = cases['ngram'][0]['frames'][1]
                tensors = cases['ngram'][1][1]
                buffers = rebuild(frame, tensors, device='cuda')
                metadata = decode_metadata(frame, tensors)
                batched = run_decode(frame, tensors, buffers, metadata)
                singles = []
                for index in range(len(frame['input_ids'])):
                    start, end = metadata['kv_indptr'][index:index+2].tolist()
                    single = {'kv_indptr': torch.tensor([0, end-start], dtype=torch.int32),
                              'kv_indices': metadata['kv_indices'][start:end],
                              'num_kv_splits': metadata['num_kv_splits'][index:index+1]}
                    singles.append(run_decode(frame, tensors, buffers, single,
                        query=buffers['q'][index:index+1]).detach())
                torch.cuda.synchronize()
                sequential = torch.cat(singles)
                result['decode_replay_batch_vs_single_queries'] = stats(batched, sequential)
                result['verify_root_using_decode_vs_control'] = stats(batched[:1], original_outputs['control'][1][:1])
                result['verify_using_decode_vs_original_verify'] = stats(batched, original_outputs['ngram'][1])
                result['same_visible_inputs'] = all(result['actual_cache_comparison']['frames'][1][key] for key in (
                    'first_query_bits_equal', 'first_query_visible_key_bits_equal', 'first_query_visible_value_bits_equal',
                    'reference_first_query_matches_causal_visibility', 'candidate_first_query_matches_causal_visibility'))
                result['route_explains_observed_root_for_this_case'] = (
                    result['same_visible_inputs'] and result['decode_replay_batch_vs_single_queries']['bit_identical']
                    and result['verify_root_using_decode_vs_control']['bit_identical']
                    and not stats(original_outputs['ngram'][1][:1], original_outputs['control'][1][:1])['bit_identical'])
                tiled, configuration = run_unified_tile(frame, tensors, buffers, 32)
                torch.cuda.synchronize()
                result['unified_tile_n32'] = {'configuration': configuration,
                    'vs_original_verify': stats(tiled, original_outputs['ngram'][1]),
                    'vs_decode_replay': stats(tiled, batched),
                    'root_vs_control': stats(tiled[:1], original_outputs['control'][1][:1])}
                result['tile_n32_restores_root_and_all_decode_rows_for_this_case'] = (
                    result['same_visible_inputs']
                    and result['decode_replay_batch_vs_single_queries']['bit_identical']
                    and result['unified_tile_n32']['vs_decode_replay']['bit_identical']
                    and result['unified_tile_n32']['root_vs_control']['bit_identical']
                    and not stats(original_outputs['ngram'][1][:1], original_outputs['control'][1][:1])['bit_identical'])
                del buffers, batched, sequential, singles, tiled
            result.update(status='completed_isolation' if reproduced else 'failed_original_reproduction',
                          gpu_replay_executed=True)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] in ('completed_isolation', 'cpu_preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
