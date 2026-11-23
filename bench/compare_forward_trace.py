#!/usr/bin/env python3
"""Compare owned bounded traces; identify differences without claiming causality."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'trace.json').read_text())
    if manifest['status'] != 'complete' or [f['phase'] for f in manifest['frames']] != ['prefill', 'step']:
        raise ValueError('Require complete prefill/first-step traces')
    tensors = []
    for frame in manifest['frames']:
        path = directory / frame['tensor_file']
        if path.resolve().parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != frame['tensor_sha256']:
            raise ValueError('Tensor artifact path or digest mismatch')
        values = torch.load(path, map_location='cpu', weights_only=True)
        keys = [leaf['key'] for e in frame['events'] for leaf in e['leaves']]
        if len(set(keys)) != len(keys) or set(keys) != set(values):
            raise ValueError('Missing/duplicate/extra captured tensors')
        for event in frame['events']:
            for leaf in event['leaves']:
                value = values[leaf['key']]
                if list(value.shape) != leaf['shape'] or str(value.dtype) != leaf['dtype']:
                    raise ValueError('Tensor metadata mismatch')
                if leaf['comparison_scope'] == 'dense_values' and value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    raise ValueError('Quantized layout cannot be labeled a dense boundary')
                if leaf['comparison_scope'] == 'dense_values' and (leaf['finite'] is not True or not torch.isfinite(value.float()).all()):
                    raise ValueError('Nonfinite capture cannot establish parity')
        tensors.append(values)
    return manifest, tensors


def compare(reference, candidate):
    a, av = read(reference)
    b, bv = read(candidate)
    if a['source_sha256'] != b['source_sha256']:
        raise ValueError('Different collector sources')
    if a['frames'][0]['input_ids'] != b['frames'][0]['input_ids']:
        raise ValueError('Different actual prefill token IDs')
    if a['frames'][0]['positions'] != b['frames'][0]['positions']:
        raise ValueError('Different actual prefill positions')
    if a['frames'][1]['input_ids'][0] != b['frames'][1]['input_ids'][0]:
        raise ValueError('First decode/verify input token differs')
    if a['frames'][1]['positions'][0] != b['frames'][1]['positions'][0]:
        raise ValueError('First decode/verify position differs')
    results = []
    for i, (af, bf) in enumerate(zip(a['frames'], b['frames'])):
        def index(frame):
            output = {}
            for event in frame['events']:
                for leaf in event['leaves']:
                    # Keys start with a per-forward event ordinal. The remaining
                    # path identifies the same leaf within a module invocation.
                    signature = (event['module'], event['stage'], leaf['key'].split(':', 1)[1])
                    if signature in output:
                        raise ValueError('Repeated module boundary needs explicit invocation alignment')
                    output[signature] = leaf
            return output
        ai, bi = index(af), index(bf)
        rows = []
        opaque = []
        for signature, x in ai.items():
            if signature not in bi:
                continue
            y = bi[signature]
            if x['comparison_scope'] != 'dense_values' or y['comparison_scope'] != 'dense_values':
                opaque.append(signature)
                continue
            va, vb = av[i][x['key']], bv[i][y['key']]
            compatible = va.dtype == vb.dtype and va.shape == vb.shape
            equal = compatible and torch.equal(va.view(torch.uint8), vb.view(torch.uint8))
            row = {'module': signature[0], 'stage': signature[1], 'path': signature[2],
                   'compatible_dense_shape_dtype': compatible, 'bit_identical': bool(equal)}
            if compatible:
                delta = (va.float() - vb.float()).abs()
                row.update(different_values=int((va.float() != vb.float()).sum()),
                           max_abs_difference=float(delta.max()) if delta.numel() else 0.,
                           numel=va.numel())
            rows.append(row)
        if not rows:
            raise ValueError('No shared dense boundaries')
        differences = [row for row in rows if not row['bit_identical']]
        results.append({'phase': af['phase'], 'reference_mode': af['mode'], 'candidate_mode': bf['mode'],
            'same_event_structure': list(ai) == list(bi),
            'reference_only_boundaries': [k for k in ai if k not in bi],
            'candidate_only_boundaries': [k for k in bi if k not in ai],
            'opaque_layouts_not_compared': opaque, 'dense_boundaries': rows,
            'first_difference_among_shared_dense_boundaries': differences[0] if differences else None,
            'shared_dense_bit_identical': not differences})
    return {'reference': str(reference), 'candidate': str(candidate), 'frames': results,
            'actual_prefill_ids_equal': True, 'actual_prefill_positions_equal': True,
            'first_step_input_id_equal': True, 'first_step_position_equal': True,
            'scope': 'Observed aligned boundaries only; missing/fused hooks and opaque quantized layouts remain explicit. A shared-boundary difference locates evidence, not a proven causal operator or full-model losslessness.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite trace comparison')
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
