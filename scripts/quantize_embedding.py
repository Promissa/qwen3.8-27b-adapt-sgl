#!/usr/bin/env python3
"""CPU NVFP4 embedding conversion for untied ModelOpt checkpoints.

Writes a separate checkpoint. Unchanged shards are symlinked; the embedding shard
is rewritten, so an iterator cannot accidentally load both BF16 and packed tensors.
Only the embedding is modified. GPU loading and quality gates remain mandatory.
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

LUT = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def quantize_rows(weight, global_scale):
    """E2M1, low nibble first; E4M3 block scales, FP32 tensor scale, block=16."""
    weight = weight.float()
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError("Expected a matrix with hidden dimension divisible by 16")
    if not torch.isfinite(weight).all() or not global_scale > 0:
        raise ValueError("Nonfinite weights or invalid global scale")
    blocks = weight.reshape(weight.shape[0], -1, 16)
    raw_scale = blocks.abs().amax(-1) / (6 * global_scale)
    # Minimum subnormal E4M3 is 2^-9. Zero blocks may safely retain a scale of 1.
    raw_scale = torch.where(raw_scale == 0, 1., raw_scale.clamp(min=2**-9, max=448))
    scales = raw_scale.to(torch.float8_e4m3fn)
    normalized = blocks / (scales.float().unsqueeze(-1) * global_scale)
    magnitudes = normalized.abs().clamp(max=6)
    # At ties use the even E2M1 code, matching round-to-nearest-even.
    levels = torch.tensor(LUT)
    distance = (magnitudes.unsqueeze(-1) - levels).abs()
    best_distance = distance.amin(-1, keepdim=True)
    tied = distance == best_distance
    priority = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15])
    codes = torch.where(tied, priority, 100).argmin(-1).to(torch.uint8)
    codes |= (normalized < 0).to(torch.uint8) << 3
    codes = codes.reshape_as(weight)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales


def update_config(config, prefix, tensor_headers):
    config = copy.deepcopy(config)
    text = config.get("text_config", config)
    if config.get("tie_word_embeddings", False) or text.get("tie_word_embeddings", False):
        raise ValueError("Tied embedding/lm_head is unsupported")
    if "tie_word_embeddings" not in config and "tie_word_embeddings" not in text:
        raise ValueError("Embedding/lm_head tying must be explicitly false")
    q = config.get("quantization_config")
    if not q or q.get("quant_method") not in ("modelopt", "modelopt_fp4", "modelopt_mixed"):
        raise ValueError("Only flat ModelOpt NVFP4/MIXED_PRECISION checkpoints are supported")
    if q.get("quant_algo") == "NVFP4":
        layers = {}
        for name, info in tensor_headers.items():
            module = name.removesuffix(".weight")
            if name.endswith(".weight") and info["dtype"] == "U8" and len(info["shape"]) == 2:
                if module + ".weight_scale" not in tensor_headers or module + ".weight_scale_2" not in tensor_headers:
                    raise ValueError(f"Packed matrix without NVFP4 scales: {name}")
                layers[module] = {"quant_algo": "NVFP4", "group_size": 16}
        if not layers:
            raise ValueError("No NVFP4 packed linear layers found")
        q.update(quant_algo="MIXED_PRECISION", quant_method="modelopt_mixed", quantized_layers=layers)
    elif q.get("quant_algo") != "MIXED_PRECISION" or not q.get("quantized_layers"):
        raise ValueError("Expected NVFP4 or a nonempty MIXED_PRECISION layer map")
    # Broad patterns may hide more than this embedding. Refuse ambiguous edits.
    for field in ("ignore", "exclude_modules"):
        remaining = []
        for pattern in q.get(field, []) or []:
            if fnmatch.fnmatchcase(prefix, pattern):
                if "embed_tokens" not in pattern:
                    raise ValueError(f"Broad exclusion {pattern!r} masks embedding; resolve explicitly")
            else:
                remaining.append(pattern)
        if field in q:
            q[field] = remaining
    q["quantized_layers"][prefix] = {"quant_algo": "NVFP4", "group_size": 16}
    # A pure NVFP4 config promoted above must always dispatch to mixed precision.
    q["quant_method"] = "modelopt_mixed"
    return config


def headers(path):
    import struct
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return {k: v for k, v in json.loads(stream.read(size)).items() if k != "__metadata__"}


def convert(source, destination, rows=256):
    source, destination = source.resolve(), destination.absolute()
    if destination.exists():
        raise ValueError("Destination must not exist; partial conversions are never treated as usable")
    if rows <= 0:
        raise ValueError("rows must be positive")
    parent_pin = source / "revision.json"
    if parent_pin.exists() and not json.loads(parent_pin.read_text()).get("complete"):
        raise ValueError("Source download/conversion is incomplete")
    shards = sorted(source.glob("*.safetensors"))
    all_headers = {}
    owners = {}
    for shard in shards:
        for key, info in headers(shard).items():
            if key in all_headers:
                raise ValueError(f"Duplicate tensor {key}")
            all_headers[key] = info
            owners[key] = shard
    embedding_keys = [k for k in all_headers if k.endswith("embed_tokens.weight")]
    if len(embedding_keys) != 1:
        raise ValueError(f"Expected one embedding, found {embedding_keys}")
    key = embedding_keys[0]
    prefix = key.removesuffix(".weight")
    if all_headers[key]["dtype"] not in ("BF16", "F16", "F32"):
        raise ValueError("Embedding is already quantized or has an unsupported dtype")
    config = update_config(json.loads((source / "config.json").read_text()), prefix, all_headers)
    owner = owners[key]
    with safe_open(owner, framework="pt", device="cpu") as f:
        sliced = f.get_slice(key)
        height, width = sliced.get_shape()
        maximum = 0.
        for start in range(0, height, rows):
            chunk = sliced[start:start + rows].float()
            if not torch.isfinite(chunk).all():
                raise ValueError("Embedding contains NaN/Inf")
            maximum = max(maximum, chunk.abs().max().item())
        global_scale = max(maximum / (6 * 448), torch.finfo(torch.float32).tiny)
        if width % 16:
            raise ValueError("Embedding hidden dimension must be divisible by 16")
        packed = torch.empty(height, width // 2, dtype=torch.uint8)
        scales = torch.empty(height, width // 16, dtype=torch.float8_e4m3fn)
        error_sum, norm_sum, max_error = 0., 0., 0.
        for start in range(0, height, rows):
            chunk = sliced[start:start + rows].float()
            p, s = quantize_rows(chunk, global_scale)
            packed[start:start + len(chunk)] = p
            scales[start:start + len(chunk)] = s
            codes = torch.empty_like(chunk, dtype=torch.uint8)
            codes[:, 0::2], codes[:, 1::2] = p & 15, p >> 4
            values = torch.tensor(LUT)[(codes & 7).long()]
            values = torch.where(codes & 8 != 0, -values, values)
            dequant = values.reshape(len(chunk), -1, 16) * s.float().unsqueeze(-1) * global_scale
            error = dequant.reshape_as(chunk) - chunk
            error_sum += error.square().sum().item()
            norm_sum += chunk.square().sum().item()
            max_error = max(max_error, error.abs().max().item())
        tensors = {name: f.get_tensor(name) for name in f.keys() if name != key}
        tensors.update({key: packed, prefix + ".weight_scale": scales,
                        prefix + ".weight_scale_2": torch.tensor([global_scale], dtype=torch.float32)})
        destination.mkdir(parents=True)
        (destination / ".conversion-in-progress").touch()
        for file in source.iterdir():
            if file == owner or file.name in ("config.json", "model.safetensors.index.json",
                                               "hf_quant_config.json", "revision.json"):
                continue
            if file.is_file():
                if file.suffix == ".safetensors":
                    (destination / file.name).symlink_to(file.resolve())
                else:
                    shutil.copy2(file, destination / file.name)
        save_file(tensors, destination / owner.name, metadata={"format": "pt"})
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    weight_map = {name: owner_path.name for name, owner_path in owners.items()}
    for suffix in (".weight_scale", ".weight_scale_2"):
        weight_map[prefix + suffix] = owner.name
    total_bytes = sum(path.stat().st_size for path in destination.glob("*.safetensors"))
    (destination / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(
            info["data_offsets"][1] - info["data_offsets"][0]
            for shard in destination.glob("*.safetensors") for info in headers(shard).values())},
        "weight_map": weight_map}, indent=2) + "\n")
    report = {"source": str(source), "source_revision": json.loads(parent_pin.read_text()) if parent_pin.exists() else None,
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "operation": "embedding-only CPU NVFP4 E2M1/E4M3 block16 RNE", "embedding": key,
        "shape": [height, width], "scale_2": global_scale,
        "relative_rmse": (error_sum / norm_sum)**.5 if norm_sum else 0.,
        "max_absolute_error": max_error, "checkpoint_file_bytes": total_bytes,
        "gpu_load_verified": False, "quality_verified": False}
    (destination / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "revision.json").write_text(json.dumps({"complete": True,
        "revision": "local-embedding-nvfp4", "parent": report["source_revision"],
        "conversion": "conversion.json"}, indent=2) + "\n")
    (destination / ".conversion-in-progress").unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(convert(args.source, args.destination, args.rows), indent=2))
