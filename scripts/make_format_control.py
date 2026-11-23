#!/usr/bin/env python3
"""Build a metadata-only control for the NVFP4 embedding experiment.

All original weight/tokenizer files are symlinked. Only config/revision metadata
changes, isolating flat-to-mixed loader dispatch from embedding quantization.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from quantize_embedding import headers, update_config


def make_control(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("Destination must not exist")
    pin = json.loads((source / "revision.json").read_text())
    config = json.loads((source / "config.json").read_text())
    original_quant = config.get("quantization_config", {})
    if not pin.get("complete") or original_quant.get("quant_algo") != "NVFP4":
        raise ValueError("Expected a complete, flat NVFP4 source checkpoint")
    info = {}
    for shard in sorted(source.glob("*.safetensors")):
        shard_info = headers(shard)
        if info.keys() & shard_info.keys():
            raise ValueError("Duplicate checkpoint tensor")
        info.update(shard_info)
    embeddings = [key for key in info if key.endswith("embed_tokens.weight")]
    if len(embeddings) != 1 or info[embeddings[0]]["dtype"] != "BF16":
        raise ValueError("Expected one unchanged BF16 embedding")
    prefix = embeddings[0].removesuffix(".weight")
    control = update_config(config, prefix, info)
    control["quantization_config"]["quantized_layers"].pop(prefix)
    for field in ("ignore", "exclude_modules"):
        if field in original_quant:
            control["quantization_config"][field] = copy.deepcopy(original_quant[field])
    destination.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.name not in ("config.json", "hf_quant_config.json", "revision.json", "conversion.json"):
            (destination / file.name).symlink_to(file.resolve())
    (destination / "config.json").write_text(json.dumps(control, indent=2) + "\n")
    report = {"operation": "flat-to-mixed metadata-only control; all weight tensors unchanged",
        "source": str(source), "source_revision": pin, "embedding": embeddings[0],
        "embedding_dtype": "BF16", "weight_files": [p.name for p in source.glob("*.safetensors")],
        "source_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_load_verified": False, "token_equality_verified": False}
    (destination / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "revision.json").write_text(json.dumps({"complete": True,
        "revision": "local-flat-to-mixed-metadata-control", "parent": pin,
        "conversion": "conversion.json"}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(make_control(args.source, args.destination), indent=2))
