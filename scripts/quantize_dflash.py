#!/usr/bin/env python3
"""Quantize DFlash2 attention/MLP weights to NVFP4 with BF16 activations.

The context projector, grouped convolutions and candidate selector stay BF16.
Their upstream modules do not use the quantization interface. No activation
calibration is required for W4A16. Runtime loading/acceptance remain live gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from quantize_embedding import LUT, headers, quantize_rows


def fused_group(name):
    return re.sub(r"\.(q|k|v)_proj$", ".qkv_proj",
                  re.sub(r"\.(gate|up)_proj$", ".gate_up_proj", name))


def quantize_matrix(sliced, shape, scale, rows):
    """Shared CPU packing for BF16-activation draft linear layers."""
    height, width = shape
    packed = torch.empty(height, width // 2, dtype=torch.uint8)
    scales = torch.empty(height, width // 16, dtype=torch.float8_e4m3fn)
    error_sum = norm_sum = 0.
    for start in range(0, height, rows):
        chunk = sliced[start:start + rows].float()
        p, s = quantize_rows(chunk, scale)
        packed[start:start + len(chunk)], scales[start:start + len(chunk)] = p, s
        codes = torch.empty_like(chunk, dtype=torch.uint8)
        codes[:, 0::2], codes[:, 1::2] = p & 15, p >> 4
        values = torch.tensor(LUT)[(codes & 7).long()]
        values = torch.where(codes & 8 != 0, -values, values)
        restored = values.reshape(len(chunk), -1, 16) * s.float().unsqueeze(-1) * scale
        error_sum += (restored.reshape_as(chunk) - chunk).square().sum().item()
        norm_sum += chunk.square().sum().item()
    return packed, scales, (error_sum / norm_sum)**.5 if norm_sum else 0.


def convert(source, destination, rows=128):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or rows <= 0:
        raise ValueError("Destination must be new and rows positive")
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["DFlash2DraftModel"] or config.get("quantization_config"):
        raise ValueError("Expected an unquantized DFlash2DraftModel")
    pin = json.loads((source / "revision.json").read_text())
    if not pin.get("complete"):
        raise ValueError("Source download is incomplete")
    owners, info = {}, {}
    for shard in sorted(source.glob("*.safetensors")):
        for key, value in headers(shard).items():
            if key in info:
                raise ValueError(f"Duplicate tensor: {key}")
            owners[key], info[key] = shard, value
    selected = {}
    for layer in range(config["num_hidden_layers"]):
        for part, names in (("self_attn", ("q", "k", "v", "o")),
                            ("mlp", ("gate", "up", "down"))):
            for name in names:
                key = f"layers.{layer}.{part}.{name}_proj.weight"
                entry = info.get(key)
                if not entry or entry["dtype"] not in ("BF16", "F16", "F32") or len(entry["shape"]) != 2 or entry["shape"][1] % 16:
                    raise ValueError(f"Missing/unsupported linear tensor: {key}")
                selected[key] = fused_group(key.removesuffix(".weight"))
    # All members of fused QKV/gate-up must use the same global scale. The
    # upstream loader takes their maximum and cannot preserve independent ones.
    maxima = {}
    for key, group in selected.items():
        maximum = 0.
        with safe_open(owners[key], framework="pt", device="cpu") as f:
            sliced = f.get_slice(key)
            for start in range(0, info[key]["shape"][0], rows):
                chunk = sliced[start:start + rows].float()
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Nonfinite tensor: {key}")
                maximum = max(maximum, chunk.abs().max().item())
        maxima[group] = max(maxima.get(group, 0.), maximum)
    global_scales = {g: max(m / (6 * 448), torch.finfo(torch.float32).tiny)
                     for g, m in maxima.items()}
    destination.mkdir(parents=True)
    (destination / ".conversion-in-progress").touch()
    reports, weight_map = {}, {}
    for shard in sorted(set(owners.values())):
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in selected:
                    tensors[key] = f.get_tensor(key)
                    continue
                scale = global_scales[selected[key]]
                height, width = info[key]["shape"]
                packed, scales, rmse = quantize_matrix(f.get_slice(key), (height, width), scale, rows)
                prefix = key.removesuffix(".weight")
                tensors.update({key: packed, prefix + ".weight_scale": scales,
                    prefix + ".weight_scale_2": torch.tensor([scale], dtype=torch.float32)})
                reports[key] = {"shape": [height, width], "fused_group": selected[key],
                    "global_scale": scale, "relative_rmse": rmse}
        save_file(tensors, destination / shard.name, metadata={"format": "pt"})
        weight_map.update({key: shard.name for key in tensors})
        print(f"Converted {shard.name}", flush=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (
                "config.json", "revision.json", "model.safetensors.index.json", "hf_quant_config.json"):
            shutil.copy2(path, destination / path.name)
    config["quantization_config"] = {"quant_method": "modelopt_fp4", "quant_algo": "W4A16_NVFP4",
        "group_size": 16, "exclude_modules": [], "kv_cache_quant_algo": "auto"}
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tensor_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0]
                       for shard in destination.glob("*.safetensors") for v in headers(shard).values())
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": tensor_bytes}, "weight_map": weight_map}, indent=2) + "\n")
    report = {"operation": "DFlash2 attention/MLP W4A16_NVFP4 CPU block16 RNE",
        "source": str(source), "source_revision": pin, "tensors": reports,
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quantizer_sha256": hashlib.sha256(Path(__file__).with_name("quantize_embedding.py").read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "checkpoint_file_bytes": sum(p.stat().st_size for p in destination.glob("*.safetensors")),
        "gpu_load_verified": False, "token_equality_verified": False, "acceptance_verified": False}
    (destination / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "revision.json").write_text(json.dumps({"complete": True,
        "revision": "local-dflash-attention-mlp-nvfp4-a16", "parent": pin,
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
