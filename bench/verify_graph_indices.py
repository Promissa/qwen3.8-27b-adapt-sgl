"""Replay shared GPU buffers for chain/tree masks, shorter prefixes and slot reuse."""

import argparse
from pathlib import Path

import torch

from common import write_json
from sglang.srt.layers.attention.triton_backend import _build_c1_verify_decode_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    n = 520
    device = "cuda"
    ptr = torch.zeros(2, dtype=torch.int32, device=device)
    prefix_ids = torch.arange(n, device=device, dtype=torch.int32)
    draft = torch.arange(4, device=device, dtype=torch.int32)
    mask = torch.zeros(4 * (n + 4), dtype=torch.uint8, device=device)
    out_ptr = torch.zeros(5, device=device, dtype=torch.int32)
    out_ids = torch.zeros(4 * (n + 4), device=device, dtype=torch.int32)
    splits = torch.zeros(4, device=device, dtype=torch.int32)

    def run():
        _build_c1_verify_decode_indices[(4, 8)](
            ptr, prefix_ids, draft, mask, out_ptr, out_ids, splits, 4, 32, 128
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    report = []
    for index, length in enumerate([0, 257, 35, 511, 8]):
        tree = torch.tril(torch.ones((4, 4), dtype=torch.bool))
        if index % 2:
            tree[2, 1] = False
            tree[3, 1] = False
        m = torch.cat([torch.ones(4, length, dtype=torch.bool), tree], dim=1)
        mask[: m.numel()].copy_(m.flatten())
        ptr[1] = length
        draft.copy_(torch.arange(4, dtype=torch.int32) + n + index)
        prefix_ids.copy_(
            torch.arange(n, dtype=torch.int32).flip(0)
            if index % 2
            else torch.arange(n, dtype=torch.int32)
        )
        graph.replay()
        all_ids = torch.cat([prefix_ids[:length].cpu(), draft.cpu()]).expand(4, -1)
        lengths = m.sum(1)
        assert torch.equal(
            out_ptr.cpu(), torch.cat([torch.zeros(1), lengths.cumsum(0)])
        )
        assert torch.equal(out_ids[: int(lengths.sum())].cpu(), all_ids[m])
        assert torch.equal(splits.cpu(), (lengths + 31) // 32)
        report.append({"prefix": length, "indices_checked": int(lengths.sum())})
    write_json(args.output, {"status": "passed", "graph_replays": report})


if __name__ == "__main__":
    main()
