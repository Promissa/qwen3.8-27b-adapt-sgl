#!/usr/bin/env python3
"""Compare actual GDN verify Conv1D against sequential single-token updates."""
import argparse
import hashlib
import inspect
from pathlib import Path
import torch
from common import now, write_json
from gdn_verify_parity import difference


def topology(parents, device):
    children = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if child:
            children[parent].append(child)
    next_token = [-1] * len(parents)
    siblings = [-1] * len(parents)
    for parent, group in enumerate(children):
        if group:
            next_token[parent] = group[0]
            for a, b in zip(group, group[1:]):
                siblings[a] = b
    return (torch.tensor([next_token], device=device, dtype=torch.int32),
            torch.tensor([siblings], device=device, dtype=torch.int32),
            torch.zeros((1, len(parents)), device=device, dtype=torch.int32))


@torch.inference_mode()
def run(output, match=False):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    device = 'cuda'
    dim = 2 * 16 * 128 + 48 * 128
    result = {'at': now(), 'status': 'running', 'device': torch.cuda.get_device_name(),
        'geometry': {'channels': dim, 'width': 4, 'state_columns': 3},
        'purpose': 'Isolate convolution output/state ancestry; not full-model equality',
        'source_sha256': hashlib.sha256(Path(inspect.getsourcefile(causal_conv1d_update)).read_bytes()).hexdigest(),
        'diagnostic_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'match_sequential_update': match, 'cases': []}
    write_json(output, result)
    try:
        for layout in ('channel_contiguous', 'window_contiguous'):
            for with_bias in (False, True):
                for mode, parents, tree in (
                    ('one_token', [-1], False),
                    ('chain', [-1, 0, 1, 2], False),
                    ('tree_chain', [-1, 0, 1, 2], True),
                    ('tree_branch', [-1, 0, 0, 2], True),
                ):
                    torch.manual_seed(1701 + len(parents) + 100 * with_bias)
                    tokens = len(parents)
                    x = torch.randn(tokens, dim, device=device).bfloat16()
                    weight = torch.randn(dim, 4, device=device).bfloat16()
                    bias = torch.randn(dim, device=device).bfloat16() if with_bias else None
                    initial = torch.randn(2, dim, 3, device=device).bfloat16()
                    # Valid finite BF16 values: canonical FP32 accumulation is
                    # ((1 + 2^-24) - 1) + 0 == 0; reversing leaves 2^-24.
                    initial[1, 0] = torch.tensor([1., 2.**-24, -1.], device=device)
                    x[0, 0] = 0
                    weight[0] = 1
                    if bias is not None:
                        bias[0] = 0
                    if layout == 'channel_contiguous':
                        initial = initial.transpose(1, 2).contiguous().transpose(1, 2)
                    state = initial.clone(memory_format=torch.preserve_format)
                    index = torch.tensor([1], device=device, dtype=torch.int32)
                    outputs, snapshots, reference = [], [], []
                    for step, parent in enumerate(parents):
                        state[1].copy_(initial[1] if parent < 0 else snapshots[parent])
                        window = torch.cat((state[1], x[step, :, None]), dim=-1)
                        acc = torch.zeros(dim, device=device) if bias is None else bias.float().clone()
                        for column in range(4):
                            acc += (window[:, column] * weight[:, column]).float()
                        reference.append((acc / (1 + torch.exp(-acc))).bfloat16())
                        out = causal_conv1d_update(x[step:step+1], state, weight, bias,
                            'silu', conv_state_indices=index)
                        outputs.append(out[0].clone())
                        snapshots.append(state[1].clone())
                        torch.testing.assert_close(state[1], window[:, 1:], rtol=0, atol=0)
                    verify_state = initial.clone(memory_format=torch.preserve_format)
                    windows = torch.empty((2, tokens, dim, 3), device=device, dtype=torch.bfloat16)
                    kwargs = {}
                    parent_tensor = None
                    if tree:
                        next_token, siblings, parent_tensor = topology(parents, device)
                        kwargs.update(retrieve_next_token=next_token, retrieve_next_sibling=siblings,
                                      retrieve_parent_token=parent_tensor)
                    if match:
                        kwargs['match_sequential_update'] = True
                    actual = causal_conv1d_update(x.unsqueeze(0).transpose(1, 2), verify_state,
                        weight, bias, 'silu', conv_state_indices=index,
                        intermediate_conv_window=windows, intermediate_state_indices=index, **kwargs)
                    actual = actual[0].transpose(0, 1)
                    expected = torch.stack(outputs)
                    states_equal = difference(windows[1], torch.stack(snapshots))
                    assert states_equal['bit_equal'], 'Convolution intermediate ancestry/state differs'
                    assert torch.equal(verify_state[0], initial[0]), 'Unrelated state slot modified'
                    if tree:
                        assert parent_tensor.tolist() == [[max(0, p) for p in parents]]
                    comparison = difference(actual, expected)
                    reference_comparison = difference(expected, torch.stack(reference))
                    assert reference_comparison['relative_rmse'] < .02
                    case = {'layout': layout, 'with_bias': with_bias, 'mode': mode,
                        'finite_cancellation_counterexample_channel': 0,
                        'parents': parents, 'verify_vs_decode_output': comparison,
                        'intermediate_states': states_equal, 'decode_vs_torch_output': reference_comparison}
                    result['cases'].append(case)
                    write_json(output, result)
                    if match:
                        assert comparison['bit_equal'], 'Opt-in convolution order does not match decode'
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-sequential-update', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new evidence file')
    run(args.output, args.match_sequential_update)
