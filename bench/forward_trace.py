"""Bounded actual Qwen forwards through SGLang's public --forward-hooks API.

Install on the backbone (target_modules=["model"]). EagerRunner calls the outer
model.forward directly, bypassing outer nn.Module hooks. Both graph phases must
be disabled. Bootstrap is deliberately unarmed during probe capacity requests;
the client arms one fresh prompt after readiness, capacity and cache flush.
This synchronizes CPU copies and is a numerical diagnostic, never a benchmark.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import torch


LAYER = re.compile(r'(?:^|\.)layers\.\d+(?:\.(?:input_layernorm|post_attention_layernorm|'
                   r'linear_attn(?:\.(?:in_proj_qkvz|in_proj_ba|out_proj|norm|attn))?|'
                   r'mlp(?:\.(?:gate_up_proj|down_proj))?|qkv_proj|o_proj|attn))?$')


class Recorder:
    def __init__(self, config):
        self.config = dict(config)
        self.arm = Path(config['arm_file'])
        self.output = Path(config['output_dir']) / f'worker-{os.getpid()}'
        self.limit = int(config.get('max_bytes', 256 * 1024**2))
        self.max_rows = int(config.get('max_prompt_rows', 1024))
        if not 0 < self.limit <= 512 * 1024**2 or not 0 < self.max_rows <= 2048:
            raise ValueError('Trace bounds must be explicit and small')
        self.used = 0
        self.current = None
        self.tensors = {}
        self.installed = False
        self.state = {'status': 'unarmed', 'frames': [], 'config': self.config,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Last prefill row and first decode/verify row; persistent GDN state after prefill and before the first step. No verify-post/persistent-decode comparison. Synchronizing diagnostic, no timing claims.'}

    def save(self):
        path = self.output / 'trace.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(path)

    def fail(self, message):
        self.state.update(status='failed', error=message)
        self.save()
        raise RuntimeError(message)

    def tensor(self, value, path, whole=False, opaque=False):
        if not isinstance(value, torch.Tensor):
            return None
        shape = list(value.shape)
        dense = not opaque and value.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        if not whole and value.ndim:
            # Row0 of the verification tree is the emitted token shared with
            # ordinary decode. In prefill, retain the final prompt position.
            index = -1 if self.current['phase'] == 'prefill' else 0
            value = value[index:index+1] if index == 0 else value[-1:]
        size = value.numel() * value.element_size()
        if self.used + size > self.limit:
            self.fail('Trace byte limit exceeded; partial trace cannot establish parity')
        self.used += size
        cpu = value.detach().to(device='cpu', copy=True).contiguous()
        key = f'{len(self.current["events"])}:{path}'
        if key in self.tensors:
            self.fail('Duplicate trace tensor key')
        self.tensors[key] = cpu
        floating = cpu.is_floating_point()
        return {'key': key, 'original_shape': shape, 'shape': list(cpu.shape),
                'dtype': str(cpu.dtype), 'bytes': size,
                'finite': bool(torch.isfinite(cpu.float()).all()) if floating else None,
                'selection_scope': 'whole' if whole else 'token_row' if dense else 'unaligned_physical_slice',
                'comparison_scope': 'dense_values' if dense else 'opaque_layout_requires_validation'}

    def value(self, value, path, whole=False, opaque=False):
        if isinstance(value, torch.Tensor):
            leaf = self.tensor(value, path, whole, opaque)
            return [leaf] if leaf else []
        if isinstance(value, (list, tuple, dict)):
            items = value.items() if isinstance(value, dict) else enumerate(value)
            items = list(items)
            # NVFP4 passes (packed_uint8, interleaved_float8_scales); other
            # quantized pairs can have FP32 scales. None has established token
            # row alignment here. Exclude the entire pair, including its scales.
            opaque = opaque or any(isinstance(item, torch.Tensor) and (
                item.dtype in (torch.uint8, torch.int8) or (
                    item.is_floating_point() and item.element_size() < 2))
                for _, item in items)
            return [leaf for key, item in items
                    for leaf in self.value(item, f'{path}.{key}', whole, opaque)]
        return []

    def event(self, name, stage, value, whole=False):
        leaves = self.value(value, 'value', whole)
        if leaves:
            self.current['events'].append({'module': name, 'stage': stage, 'leaves': leaves})

    def state_event(self, name, module, stage):
        from sglang.srt.model_executor.forward_context import get_attn_backend
        backend = get_attn_backend()
        backend = getattr(backend, 'linear_attn_backend', backend)
        indices = backend.forward_metadata.mamba_cache_indices
        if indices.numel() != 1:
            self.fail('GDN trace requires exactly one request state slot')
        index = int(indices.item())
        cache = backend.state_layer_cache(module.layer_id)
        self.event(name, stage, {'conv': cache.conv[0][index],
                                'temporal': cache.temporal[index]}, whole=True)

    def begin(self, module, args, kwargs):
        if len(self.state['frames']) >= 2 or not self.arm.exists():
            return
        if self.current is not None:
            self.fail('Nested/reentrant model forward is unsupported')
        batch = kwargs.get('forward_batch')
        if batch is None:
            batch = next((x for x in args if hasattr(x, 'forward_mode') and hasattr(x, 'input_ids')), None)
        if batch is None:
            self.fail('Cannot identify actual ForwardBatch')
        mode = batch.forward_mode
        phase = 'step' if mode.is_decode() or mode.is_target_verify() else 'prefill'
        expected = 'prefill' if not self.state['frames'] else 'step'
        if phase != expected:
            self.fail(f'Expected one prefill then first step, got {phase} at frame{len(self.state["frames"])}')
        ids = batch.input_ids.detach().cpu().tolist()
        if not ids or len(ids) > self.max_rows or getattr(batch, 'batch_size', 1) != 1:
            self.fail('Require one short nonempty prompt, no chunking or concurrent requests')
        positions = kwargs.get('positions', args[1] if len(args) > 1 else None)
        if not isinstance(positions, torch.Tensor) or positions.ndim != 1 or positions.numel() != len(ids):
            self.fail('Require explicit one-dimensional positions aligned with actual token IDs')
        self.current = {'index': len(self.state['frames']), 'phase': phase,
            'mode': getattr(mode, 'name', str(mode)), 'input_ids': ids,
            'positions': positions.detach().cpu().tolist(),
            'seq_lens': batch.seq_lens.detach().cpu().tolist(), 'events': []}
        self.tensors = {}
        self.state['status'] = 'capturing'

    def before(self, name, module, args, kwargs):
        if self.current is None:
            return
        values = {'args': args, 'kwargs': {k: v for k, v in kwargs.items()
                                         if k in ('hidden_states', 'positions')}}
        self.event(name, 'input', values)
        if name.endswith('.linear_attn') and self.current['phase'] == 'step':
            self.state_event(name, module, 'persistent_before_step')

    def after(self, name, module, args, output):
        if self.current is None:
            return
        self.event(name, 'output', output)
        if name.endswith('.linear_attn') and self.current['phase'] == 'prefill':
            self.state_event(name, module, 'persistent_after_prefill')

    def finish(self, module, args, output):
        if self.current is None:
            return
        # Actual target backbone returns final normalized hidden states. The
        # outer logits processor runs after this boundary and is not observed.
        self.event('<backbone>', 'output', getattr(output, 'next_token_logits', output))
        filename = f'forward-{self.current["index"]:02d}.pt'
        path = self.output / filename
        torch.save(self.tensors, path)
        self.current.update(tensor_file=filename,
            tensor_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.state['frames'].append(self.current)
        self.state.update(status='complete' if len(self.state['frames']) == 2 else 'awaiting_first_step',
                          captured_tensor_bytes=self.used)
        self.current = None
        self.tensors = {}
        self.save()

    def bootstrap(self, module, args, output):
        if self.installed:
            return
        self.installed = True
        expected = self.config.get('expected_model_class', 'Qwen3_5ForCausalLM')
        if type(module).__name__ != expected:
            return  # Draft workers are separate models; capture target only.
        self.output.mkdir(parents=True, exist_ok=False)
        names = []
        for name, child in module.named_modules():
            if LAYER.search(name) or name in ('embed_tokens', 'norm', 'model.embed_tokens', 'model.norm'):
                logical_name = self.config.get('module_prefix', 'model') + '.' + name
                names.append(logical_name)
                child.register_forward_pre_hook(
                    lambda m, a, k, n=logical_name: self.before(n, m, a, k), with_kwargs=True)
                child.register_forward_hook(lambda m, a, o, n=logical_name: self.after(n, m, a, o))
        self.state.update(model_class=type(module).__name__, modules=names)
        if not names:
            self.fail('No model modules matched; no diagnostic evidence')
        module.register_forward_pre_hook(self.begin, with_kwargs=True)
        module.register_forward_hook(self.finish)
        self.save()


def make_hook(config):
    return Recorder(config).bootstrap
