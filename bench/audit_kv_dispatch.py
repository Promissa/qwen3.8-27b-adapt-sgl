#!/usr/bin/env python3
"""CPU audit of KV quantization dispatch and absent-scale initialization."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine/sglang/python'))
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config, ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.radix_attention import RadixAttention


def audit(name):
    directory = ROOT / 'weights' / name
    raw = (directory / 'config.json').read_bytes()
    config = json.loads(raw)
    quant = config['quantization_config']
    cls = ModelOptMixedPrecisionConfig if quant['quant_method'] == 'modelopt_mixed' else ModelOptFp4Config
    quant_config = cls.from_config(quant)
    text = config.get('text_config', config)
    names = []
    for shard in sorted(directory.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as tensors:
            names.extend(tensors.keys())
    # Scales would require real checkpoint loading; do not silently test defaults
    # if a checkpoint actually contains calibrated values.
    scale_names = sorted(n for n in names if n.endswith(('.k_scale', '.v_scale', '.kv_scale')))
    if scale_names:
        raise ValueError(f'Checkpoint contains KV scales requiring load validation: {scale_names}')
    layers = []
    for index, kind in enumerate(text['layer_types']):
        if kind != 'full_attention':
            continue
        prefix = f'model.language_model.layers.{index}.self_attn.attn'
        with torch.device('cpu'):
            layer = RadixAttention(text['num_attention_heads'], text['head_dim'],
                text['head_dim'] ** -.5, num_kv_heads=text['num_key_value_heads'],
                layer_id=index, prefix=prefix, quant_config=quant_config)
            method = layer.quant_method
            initial = [layer.k_scale.item(), layer.v_scale.item()]
            method.process_weights_after_loading(layer)
        layers.append({'layer': index, 'prefix': prefix,
            'method': type(method).__name__, 'kv_cache_quant_algo': quant_config.kv_cache_quant_algo,
            'initial_scales': initial, 'final_scales': [layer.k_scale.item(), layer.v_scale.item()],
            'float_scales': [layer.k_scale_float, layer.v_scale_float],
            'device': str(layer.k_scale.device), 'dtype': str(layer.k_scale.dtype)})
    if len(layers) != 16:
        raise ValueError('Expected all16 full-attention layers')
    return {'config_sha256': hashlib.sha256(raw).hexdigest(),
            'config_class': cls.__name__, 'scale_tensor_names': scale_names, 'layers': layers}


if __name__ == '__main__':
    output = ROOT / 'results/mixed-format-kv-dispatch-audit.json'
    if output.exists():
        raise FileExistsError(output)
    models = {name: audit(name) for name in ('target', 'target-mixed-control', 'target-embedding-nvfp4')}
    expected = models['target']['layers']
    result = {'scope': 'CPU real RadixAttention construction and no-checkpoint-scale post-load path; no full-model GPU equivalence proof',
        'models': models, 'all_methods_and_default_scales_identical': all(x['layers'] == expected for x in models.values()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'result': str(output), 'identical': result['all_methods_and_default_scales_identical'],
                      'layers': sum(len(x['layers']) for x in models.values())}))
