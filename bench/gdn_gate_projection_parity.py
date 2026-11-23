#!/usr/bin/env python3
"""Compare actual BF16 GDN gate projections across decode/prefill/verify row counts."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

from common import ROOT, now, write_json


def run(destination):
    import torch
    from safetensors import safe_open
    from sglang.srt.batch_invariant_ops import set_batch_invariant_mode
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod, get_bf16_gemm_backend, initialize_bf16_gemm_config)
    from sglang.srt.runtime_context import get_context

    checkpoint = ROOT / 'weights/target'
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    result = {'at': now(), 'status': 'running', 'cases': [], 'device': torch.cuda.get_device_name(),
        'target_revision': json.loads((checkpoint / 'revision.json').read_text()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method_source_sha256': hashlib.sha256(Path(inspect.getsourcefile(UnquantizedLinearMethod)).read_bytes()).hexdigest(),
        'scope': 'Four actual GDN in_proj_ba weights, native UnquantizedLinearMethod and synthetic BF16 activations. Exact operator row-batching test, not full-model losslessness or representative inference-error rates.'}
    write_json(destination, result)
    start = time.monotonic()
    try:
        with get_context().override_server_args(), torch.inference_mode():
            initialize_bf16_gemm_config()
            result['bf16_gemm_backend'] = get_bf16_gemm_backend().value
            method = UnquantizedLinearMethod()
            for layer_id in (0, 1, 30, 62):
                keys = [f'model.language_model.layers.{layer_id}.linear_attn.in_proj_{part}.weight' for part in ('b', 'a')]
                weights = []
                for key in keys:
                    with safe_open(checkpoint / index[key], framework='pt', device='cpu') as handle:
                        weights.append(handle.get_tensor(key))
                widths = [weight.shape[0] for weight in weights]
                k = weights[0].shape[1]
                with torch.device('cuda'):
                    layer = torch.nn.Module()
                    method.create_weights(layer, input_size_per_partition=k, output_partition_sizes=widths,
                        input_size=k, output_size=sum(widths), params_dtype=torch.bfloat16)
                    layer.weight.copy_(torch.cat(weights, dim=0))
                    method.process_weights_after_loading(layer)
                torch.manual_seed(1701 + layer_id)
                source = torch.randn(1024, k, device='cuda', dtype=torch.bfloat16)
                for magnitude in (.125, 1.):
                    x = source * magnitude
                    for invariant in (False, True):
                        with set_batch_invariant_mode(invariant):
                            sequential = torch.cat([method.apply(layer, x[i:i + 1]) for i in range(len(x))], dim=0)
                            outputs = [(rows, method.apply(layer, x[:rows])) for rows in (1, 2, 4, 8, 32, 128, 1024)]
                        for rows, output in outputs:
                            reference = sequential[:rows]
                            finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
                            if not finite:
                                raise RuntimeError('Nonfinite gate projection')
                            result['cases'].append({'layer': layer_id, 'checkpoint_keys': keys, 'n': sum(widths), 'k': k,
                                'rows': rows, 'activation_magnitude': magnitude, 'aten_batch_invariant': invariant,
                                'finite': finite, 'exact': torch.equal(output, reference),
                                'different_elements': int((output != reference).sum().item()), 'elements': output.numel(),
                                'maximum_absolute_difference': float((output.float() - reference.float()).abs().max().item())})
                        write_json(destination, result)
                del layer, source, x, sequential, outputs, output, reference, weights
                torch.cuda.empty_cache()
            result.update(status='completed', cases_count=len(result['cases']),
                exact_cases=sum(case['exact'] for case in result['cases']), elapsed_s=time.monotonic() - start)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination, result)
    print(json.dumps({'status': result['status'], 'cases': result['cases_count'],
        'exact_by_mode': {str(mode): sum(case['exact'] for case in result['cases'] if case['aten_batch_invariant'] == mode)
                          for mode in (False, True)}, 'elapsed_s': result['elapsed_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new evidence file')
    run(args.output)
