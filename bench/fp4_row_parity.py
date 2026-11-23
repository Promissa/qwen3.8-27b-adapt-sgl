#!/usr/bin/env python3
"""Actual checkpoint/ModelOpt linears: compare batched rows to sequential decode."""
import argparse
from contextlib import nullcontext
import gc
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
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize, initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod
    from sglang.srt.runtime_context import get_context, get_exec

    checkpoint = ROOT / 'weights/target'
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    config = json.loads((checkpoint / 'config.json').read_text())['quantization_config']
    gdn = 'model.language_model.layers.0.linear_attn.'
    mlp = 'model.language_model.layers.0.mlp.'
    attention = 'model.language_model.layers.3.self_attn.'
    groups = [('gdn_qkv', [gdn + 'in_proj_qkv']), ('gdn_z', [gdn + 'in_proj_z']),
              ('gdn_out', [gdn + 'out_proj']), ('mlp_gate_up', [mlp + 'gate_proj', mlp + 'up_proj']),
              ('mlp_down', [mlp + 'down_proj']),
              ('attention_qkv', [attention + part + '_proj' for part in ('q', 'k', 'v')]),
              ('attention_out', [attention + 'o_proj']), ('head', ['lm_head'])]
    result = {'at': now(), 'status': 'running', 'cases': [], 'modules': [],
        'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
        'target_revision': json.loads((checkpoint / 'revision.json').read_text()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'method_source_sha256': hashlib.sha256(Path(inspect.getsourcefile(ModelOptFp4LinearMethod)).read_bytes()).hexdigest(),
        'scope': 'Actual checkpoint bytes through native ModelOpt weight preparation/apply with synthetic BF16 activations. Covers ordinary FP4 linear row batching, not fused SwiGLU/proj-conv, a complete forward pass, inference-error frequency or speed.'}
    write_json(destination, result)

    def tensor(key):
        with safe_open(checkpoint / index[key], framework='pt', device='cpu') as handle:
            return handle.get_tensor(key)

    def plain_scales(swizzled, rows, columns):
        rp, cp = (rows + 127) // 128 * 128, (columns + 3) // 4 * 4
        return swizzled.view(torch.uint8).reshape(1, rp // 128, cp // 4, 32, 4, 4).permute(
            0, 1, 4, 3, 2, 5).contiguous().reshape(rp, cp)[:rows, :columns]

    def difference(left, right):
        delta = (left.float() - right.float()).abs()
        return {'exact': torch.equal(left, right), 'different_elements': int((left != right).sum().item()),
                'elements': left.numel(), 'maximum_absolute_difference': float(delta.max().item()),
                'finite': bool(torch.isfinite(left).all() and torch.isfinite(right).all())}

    start = time.monotonic()
    try:
        with get_context().override_server_args(), get_exec().kernel.override(fp4_gemm_runner_backend='flashinfer_cutlass'):
            initialize_fp4_gemm_config()
            quantization = ModelOptFp4Config.from_config(config)
            for module_index, (label, prefixes) in enumerate(groups):
                weights = [tensor(prefix + '.weight') for prefix in prefixes]
                widths = [weight.shape[0] for weight in weights]
                k = weights[0].shape[1] * 2
                assert all(weight.shape[1] * 2 == k for weight in weights)
                with torch.device('cuda'):
                    layer = torch.nn.Module()
                    method = ModelOptFp4LinearMethod(quantization)
                    assert method.quant_mode == 'w4a4'
                    method.create_weights(layer, input_size_per_partition=k, output_partition_sizes=widths,
                        input_size=k, output_size=sum(widths), params_dtype=torch.bfloat16)
                    for suffix in ('weight', 'weight_scale', 'input_scale', 'weight_scale_2'):
                        values = weights if suffix == 'weight' else [tensor(prefix + '.' + suffix) for prefix in prefixes]
                        if suffix in ('input_scale', 'weight_scale_2'):
                            value = torch.cat([item.reshape(-1) for item in values])
                        else:
                            value = values[0] if len(values) == 1 else torch.cat(values, dim=0)
                        with torch.no_grad():
                            getattr(layer, suffix).copy_(value.reshape_as(getattr(layer, suffix)))
                    method.process_weights_after_loading(layer)
                del weights, values, value
                result['modules'].append({'label': label, 'checkpoint_prefixes': prefixes, 'n': sum(widths), 'k': k,
                                          'method': type(method).__name__, 'quant_mode': method.quant_mode})
                torch.manual_seed(1701 + module_index)
                source = torch.randn(32, k, device='cuda', dtype=torch.bfloat16)
                for magnitude in (.125, 1.):
                    x = source * magnitude
                    for invariant in (False, True):
                        # Use each mode's own sequential control. The ATen
                        # registration does not itself replace custom FP4 ops.
                        with torch.inference_mode(), set_batch_invariant_mode(invariant):
                            sequential = torch.cat([method.apply(layer, x[i:i + 1]) for i in range(32)], dim=0)
                            packed = [fp4_quantize(x[i:i + 1], layer.input_scale_inv) for i in range(32)]
                            sequential_q = torch.cat([item[0] for item in packed], dim=0)
                            sequential_sf = torch.cat([plain_scales(item[1], 1, k // 16) for item in packed], dim=0)
                            outputs = []
                            for rows in (1, 2, 4, 8, 32):
                                q, sf = fp4_quantize(x[:rows], layer.input_scale_inv)
                                raw_q, raw_sf = fp4_quantize(x[:rows], layer.input_scale_inv, is_sf_swizzled_layout=False)
                                logical_sf = plain_scales(sf, rows, k // 16)
                                assert torch.equal(q, raw_q) and torch.equal(logical_sf, raw_sf.view(torch.uint8).reshape(rows, k // 16)), 'Scale inverse must match the actual unswizzled quantizer'
                                output = method.apply(layer, x[:rows])
                                outputs.append((rows, output, torch.equal(q, sequential_q[:rows]),
                                                torch.equal(logical_sf, sequential_sf[:rows])))
                        for rows, output, packed_exact, scale_exact in outputs:
                            comparison = difference(output, sequential[:rows])
                            if not comparison['finite']:
                                raise RuntimeError('Nonfinite diagnostic output')
                            result['cases'].append({'module': label, 'rows': rows, 'activation_magnitude': magnitude,
                                'aten_batch_invariant': invariant, 'packed_activation_exact': packed_exact,
                                'logical_activation_scales_exact': scale_exact, 'output': comparison})
                        write_json(destination, result)
                del layer, method, source, x, sequential, packed, sequential_q, sequential_sf, outputs, output
                gc.collect()
                torch.cuda.empty_cache()
            result.update(status='completed', cases_count=len(result['cases']),
                exact_output_cases=sum(case['output']['exact'] for case in result['cases']),
                exact_packed_activation_cases=sum(case['packed_activation_exact'] for case in result['cases']),
                exact_activation_scale_cases=sum(case['logical_activation_scales_exact'] for case in result['cases']),
                elapsed_s=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination, result)
    print(json.dumps({key: result[key] for key in ('status', 'cases_count', 'exact_output_cases', 'exact_packed_activation_cases', 'exact_activation_scale_cases', 'elapsed_s')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new evidence file')
    run(args.output)
