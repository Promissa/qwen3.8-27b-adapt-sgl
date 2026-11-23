#!/usr/bin/env python3
"""Compare actual GDN verification against sequential packed decode.

Uses Qwen3.8-27B's H=16, HV=48, K=V=128, independently of GEMMs, convolution,
token sampling and the full model. Diagnostics do not change engine behavior.
"""
import argparse
import hashlib
import inspect
from pathlib import Path

import torch
from common import now, write_json


def difference(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = actual.float() - expected.float()
    bits = torch.int16 if actual.element_size() == 2 else torch.int32
    return {"bit_equal": torch.equal(actual.view(bits), expected.view(bits)),
        "different_elements": int((actual.view(bits) != expected.view(bits)).sum()),
        "elements": actual.numel(), "max_absolute_error": error.abs().max().item(),
        "relative_rmse": (error.square().sum() / expected.float().square().sum().clamp_min(1e-30)).sqrt().item()}


def torch_reference(q, k, v, a, b, log_a, dt_bias, initial, parents, state_dtype):
    """Independent V×K recurrence, including packed decode's gate/state casts."""
    outputs, snapshots = [], []
    for i in range(q.shape[1]):
        state = initial.float().clone() if i == 0 else snapshots[parents[i]].float().clone()
        # Three value heads share each query/key head.
        qi, ki = q[0, i].float().repeat_interleave(3, 0), k[0, i].float().repeat_interleave(3, 0)
        qi = qi / (qi.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ki = ki / (ki.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        qi *= q.shape[-1] ** -.5
        gate = -log_a.float().exp() * torch.nn.functional.softplus(a[i].float() + dt_bias.float())
        beta = b[i].float().sigmoid().to(b.dtype).float()
        state *= gate.exp()[:, None, None]
        delta = (v[0, i].float() - (state * ki[:, None, :]).sum(-1)) * beta[:, None]
        state += delta[:, :, None] * ki[:, None, :]
        outputs.append((state * qi[:, None, :]).sum(-1).to(v.dtype))
        snapshots.append(state.to(state_dtype))
    return torch.stack(outputs).unsqueeze(0), torch.stack(snapshots)


@torch.inference_mode()
def run(output, match_packed=False):
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
    torch.manual_seed(1701)
    torch.cuda.set_device(0)
    h, hv, kdim, vdim = 16, 48, 128, 128
    result = {"at": now(), "status": "running", "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "source_files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(inspect.getsourcefile(fused_recurrent_gated_delta_rule_packed_decode)),
                         Path(inspect.getsourcefile(fused_sigmoid_gating_delta_rule_update)))},
        "geometry": {"key_heads": h, "value_heads": hv, "key_dim": kdim, "value_dim": vdim},
        "purpose": "isolate recurrence arithmetic; not full-model token equality", "cases": []}
    result["match_packed_gdn_decode"] = match_packed
    result["diagnostic_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output, result)
    try:
        for state_dtype in (torch.float32, torch.bfloat16):
            for gate_dtype in (torch.bfloat16, torch.float32):
                for mode, parent_list in (
                    ("one_token", [0]),
                    ("chain", [0, 0, 1, 2]),
                    ("tree_chain", [0, 0, 1, 2]),
                    ("tree_branch", [0, 0, 0, 2]),
                ):
                    tokens = len(parent_list)
                    # Identical inputs across dtype cases. FP32 gate inputs are
                    # exact upcasts of the same BF16 BA values, isolating the
                    # packed kernel's sigmoid-output cast from input precision.
                    torch.manual_seed(1701 + tokens + (1000 if parent_list == [0, 0, 0, 2] else 0))
                    q = torch.randn(1, tokens, h, kdim, device="cuda", dtype=torch.bfloat16)*.2
                    k = torch.randn_like(q)*.2
                    v = torch.randn(1, tokens, hv, vdim, device="cuda", dtype=torch.bfloat16)*.2
                    a = (torch.randn(tokens, hv, device="cuda")*.5).bfloat16().to(gate_dtype)
                    b = (torch.randn(tokens, hv, device="cuda")*.5).bfloat16().to(gate_dtype)
                    log_a = torch.randn(hv, device="cuda")*.2
                    dt_bias = torch.full((hv,), -2., device="cuda")
                    initial = (torch.randn(2, hv, vdim, kdim, device="cuda")*.05).to(state_dtype)
                    sequential = initial.clone()
                    index = torch.tensor([1], device="cuda", dtype=torch.int32)
                    actual_outputs, actual_states = [], []
                    for i in range(tokens):
                        sequential[1].copy_(initial[1] if i == 0 else actual_states[parent_list[i]])
                        packed = torch.cat((q[:, i].flatten(1), k[:, i].flatten(1), v[:, i].flatten(1)), -1)
                        step_out = torch.empty((1, 1, hv, vdim), device="cuda", dtype=v.dtype)
                        fused_recurrent_gated_delta_rule_packed_decode(packed, a[i:i+1], b[i:i+1],
                            log_a, dt_bias, kdim**-.5, sequential, step_out, index, True)
                        actual_outputs.append(step_out[:, 0].clone())
                        actual_states.append(sequential[1].clone())
                    decode_output = torch.stack(actual_outputs, 1)
                    decode_states = torch.stack(actual_states)
                    verify_initial = initial.clone()
                    intermediate = torch.zeros(2, tokens, hv, vdim, kdim, device="cuda", dtype=state_dtype)
                    # None is the linear-chain path; the explicit parent table
                    # also tests the tree path used by NGRAM.
                    # The same linear sequence is also run through the tree
                    # path, isolating its typed-parent reload from the chain's
                    # FP32 accumulator without changing inputs or ancestry.
                    tree = mode.startswith("tree_")
                    parents = torch.tensor([parent_list], device="cuda", dtype=torch.int64) if tree else None
                    verify_output = fused_sigmoid_gating_delta_rule_update(
                        A_log=log_a, a=a, dt_bias=dt_bias, softplus_beta=1., softplus_threshold=20.,
                        q=q, k=k, v=v, b=b, initial_state_source=verify_initial,
                        initial_state_indices=index, use_qk_l2norm_in_kernel=True,
                        cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
                        disable_state_update=True, intermediate_states_buffer=intermediate,
                        intermediate_state_indices=index, cache_steps=tokens, retrieve_parent_token=parents,
                        match_packed_gdn_decode=match_packed)
                    assert torch.equal(verify_initial, initial), "Verification mutated persistent state"
                    ref_out, ref_states = torch_reference(q, k, v, a, b, log_a, dt_bias, initial[1], parent_list, state_dtype)
                    reference_error = difference(decode_output, ref_out)
                    assert reference_error["relative_rmse"] < .02, "Packed reference/layout disagreement"
                    reference_state_error = difference(decode_states, ref_states)
                    assert reference_state_error["relative_rmse"] < .02, "Packed state/reference disagreement"
                    row = {"mode": mode, "state_dtype": str(state_dtype), "gate_dtype": str(gate_dtype),
                        "tokens": tokens, "tree": tree, "parents": parent_list,
                        "verify_vs_decode_output": difference(verify_output, decode_output),
                        "verify_vs_decode_states": difference(intermediate[1], decode_states),
                        "decode_vs_torch_output": reference_error,
                        "decode_vs_torch_states": reference_state_error}
                    result["cases"].append(row)
                    write_json(output, result)
                    if match_packed:
                        assert row["verify_vs_decode_output"]["bit_equal"], "Verification output differs from packed decode"
                        assert row["verify_vs_decode_states"]["bit_equal"], "Verification state differs from packed decode"
        result.update(status="completed", peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        write_json(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-packed-gdn-decode", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence file")
    run(args.output, args.match_packed_gdn_decode)
