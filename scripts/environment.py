#!/usr/bin/env python3
"""Record observable environment; never infer GPU access from installed packages."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"command": argv, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": argv, "returncode": None, "error": str(e)}


def collect():
    versions = {}
    for name in ("sglang", "torch", "flashinfer-python", "flashinfer-cubin",
                 "sgl-kernel", "transformers", "huggingface-hub", "safetensors", "sgl-eval", "socksio"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source = Path(os.environ.get("SGLANG_SOURCE", "~/projects/sglang")).expanduser()
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions,
        "wsl_gpu_device_present": Path("/dev/dxg").exists(),
        "nvidia_smi": command(["nvidia-smi"]), "nvcc": command(["nvcc", "--version"]),
        "torch_cuda": command([sys.executable, "-c", "import torch; print({'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"]),
        "memory": command(["free", "-b"]), "filesystem": command(["df", "-T", "."]),
        "source_revision": command(["git", "-C", str(source), "rev-parse", "HEAD"]),
        "source_changes": command(["git", "-C", str(source), "status", "--short"]),
        "experiment_environment": {k: v for k, v in os.environ.items()
                                   if k.startswith(("SGLANG_", "FLASHINFER_")) or k in (
                                       "MAX_JOBS", "CUDA_HOME", "FLASHINFER_WORKSPACE_BASE",
                                       "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        "manual_checks": {"sysmem_fallback_disabled": "unverified",
                          "display_on_igpu": "unverified", "wsl_memory_limit": "unverified"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2) + "\n")
    print(args.output)
