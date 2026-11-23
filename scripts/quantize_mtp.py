#!/usr/bin/env python3
"""Convert the dense reference MTP attention/MLP to NVFP4 with BF16 activations.

The backbone, embedding, head, MTP fc and norms retain their original tensors.
Only affected shards are rewritten. This prepares the conditional MTP experiment;
GPU load, inference, losslessness and speed remain separate gates.
"""
import argparse
import copy
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from quantize_dflash import fused_group, quantize_matrix
from quantize_embedding import headers


def convert(source, destination, rows=128):
    source, destination = source.resolve(), destination.absolute()
    if destination.exists() or rows <= 0:
        raise ValueError("Destination must be new and rows positive")
    pin = json.loads((source / "revision.json").read_text())
    if not pin.get("complete"):
        raise ValueError("Source checkpoint is incomplete")
    config = copy.deepcopy(json.loads((source / "config.json").read_text()))
    quant = config.get("quantization_config", {})
    if (quant.get("quant_algo") != "MIXED_PRECISION" or not quant.get("quantized_layers")
            or quant.get("quant_method") not in ("modelopt", "modelopt_mixed")):
        raise ValueError("Expected a mixed ModelOpt reference checkpoint")
    info, owners = {}, {}
    for shard in sorted(source.glob("*.safetensors")):
        for key, entry in headers(shard).items():
            if key in info:
                raise ValueError(f"Duplicate tensor: {key}")
            info[key], owners[key] = entry, shard
    selected = {}
    for part, names in (("self_attn", ("q", "k", "v", "o")),
                        ("mlp", ("gate", "up", "down"))):
        for name in names:
            key = f"mtp.layers.0.{part}.{name}_proj.weight"
            entry = info.get(key)
            if (not entry or entry["dtype"] not in ("BF16", "F16", "F32")
                    or len(entry["shape"]) != 2 or entry["shape"][1] % 16):
                raise ValueError(f"Missing/unsupported dense MTP tensor: {key}")
            selected[key] = fused_group(key.removesuffix(".weight"))
    if any(k.startswith("mtp.layers.") and not k.startswith("mtp.layers.0.") for k in info):
        raise ValueError("Only the single-layer dense MTP layout is supported")
    for field in ("ignore", "exclude_modules"):
        remaining = []
        for pattern in quant.get(field, []) or []:
            if any(fnmatch.fnmatchcase(k.removesuffix(".weight"), pattern) for k in selected):
                if not pattern.startswith("mtp"):
                    raise ValueError(f"Broad exclusion masks MTP: {pattern}")
            else:
                remaining.append(pattern)
        if field in quant:
            quant[field] = remaining
    quant["quant_method"] = "modelopt_mixed"
    for key in selected:
        quant["quantized_layers"][key.removesuffix(".weight")] = {
            "quant_algo": "W4A16_NVFP4", "group_size": 16}
    if "config_groups" in quant:
        quant["config_groups"]["mtp_w4a16"] = {
            "weights": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
            "input_activations": None, "targets": [k.removesuffix(".weight") for k in selected]}
    maxima = {}
    for key, group in selected.items():
        maximum = 0.
        with safe_open(owners[key], framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(key)
            for start in range(0, info[key]["shape"][0], rows):
                chunk = sliced[start:start + rows].float()
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite tensor: {key}")
                maximum = max(maximum, chunk.abs().max().item())
        maxima[group] = max(maxima.get(group, 0.), maximum)
    scales = {g: max(m / (6 * 448), torch.finfo(torch.float32).tiny) for g, m in maxima.items()}
    destination.mkdir(parents=True)
    (destination / ".conversion-in-progress").touch()
    affected = {owners[key] for key in selected}
    reports, weight_map = {}, {key: owner.name for key, owner in owners.items()}
    for shard in sorted(set(owners.values())):
        if shard not in affected:
            (destination / shard.name).symlink_to(shard)
            continue
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key not in selected:
                    tensors[key] = handle.get_tensor(key)
                    continue
                scale = scales[selected[key]]
                packed, local, rmse = quantize_matrix(handle.get_slice(key), info[key]["shape"], scale, rows)
                prefix = key.removesuffix(".weight")
                tensors.update({key: packed, prefix + ".weight_scale": local,
                    prefix + ".weight_scale_2": torch.tensor([scale], dtype=torch.float32)})
                for suffix in (".weight_scale", ".weight_scale_2"):
                    weight_map[prefix + suffix] = shard.name
                reports[key] = {"shape": info[key]["shape"], "fused_group": selected[key],
                    "global_scale": scale, "relative_rmse": rmse}
        save_file(tensors, destination / shard.name, metadata={"format": "pt"})
        print(f"Converted {shard.name}", flush=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (
                "config.json", "revision.json", "model.safetensors.index.json", "hf_quant_config.json", "conversion.json"):
            shutil.copy2(path, destination / path.name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    output_headers = {k: entry for shard in destination.glob("*.safetensors") for k, entry in headers(shard).items()}
    size = lambda entry: entry["data_offsets"][1] - entry["data_offsets"][0]
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(map(size, output_headers.values()))},
        "weight_map": weight_map}, indent=2) + "\n")
    report = {"operation": "dense MTP attention/MLP W4A16_NVFP4 CPU block16 RNE",
        "source": str(source), "source_revision": pin, "tensors": reports,
        "source_mtp_tensor_bytes": sum(size(v) for k, v in info.items() if k.startswith("mtp.")),
        "converted_mtp_tensor_bytes": sum(size(v) for k, v in output_headers.items() if k.startswith("mtp.")),
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name("quantize_dflash.py"),
            Path(__file__).with_name("quantize_embedding.py"), source / "config.json")},
        "backbone_modified": False, "gpu_load_verified": False,
        "token_equality_verified": False, "acceptance_verified": False}
    (destination / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "revision.json").write_text(json.dumps({"complete": True,
        "revision": "local-reference-mtp-nvfp4-a16", "parent": pin,
        "conversion": "conversion.json"}, indent=2) + "\n")
    (destination / ".conversion-in-progress").unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(convert(args.source, args.destination, args.rows), indent=2))
