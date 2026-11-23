#!/usr/bin/env python3
"""Assemble a workspace-only CUDA root from split local/shared wheel installs."""
import importlib.metadata
from pathlib import Path


def prepare():
    root = Path(__file__).resolve().parents[1]
    compiler = importlib.metadata.distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13").resolve()
    cublas = importlib.metadata.distribution("nvidia-cublas").locate_file("nvidia/cu13").resolve()
    sources = list(dict.fromkeys([compiler, cublas]))
    if len(sources) == 1:
        return compiler
    # Local compiler/runtime/CRT/NVVM take precedence. Shared CUDA 13.0 cuBLAS
    # headers/libs supply the pieces not installed in this small overlay venv.
    destination = root / ".cache/cuda-toolkit"
    selected = set()
    for source in sources:
        for file in source.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(source)
            if relative in selected:
                continue
            selected.add(relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() == file.resolve():
                    continue
                target.unlink()
            elif target.exists():
                raise RuntimeError(f"Refusing to replace a non-symlink CUDA overlay file: {target}")
            target.symlink_to(file.resolve())
    lib64 = destination / "lib64"
    if not lib64.exists():
        lib64.symlink_to("lib", target_is_directory=True)
    # Wheels ship versioned sonames; FlashInfer's JIT links with -l<name>.
    for name in ("cudart", "cublas", "cublasLt"):
        runtime = destination / f"lib/lib{name}.so.13"
        unversioned = destination / f"lib/lib{name}.so"
        if runtime.exists() and not unversioned.exists():
            if unversioned.is_symlink():
                unversioned.unlink()
            unversioned.symlink_to(runtime.name)
    # The WSL driver library is supplied by Windows rather than CUDA wheels.
    driver = Path("/usr/lib/wsl/lib/libcuda.so.1")
    stub = destination / "lib/stubs/libcuda.so"
    if driver.exists() and not stub.exists():
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.symlink_to(driver)
    return destination


if __name__ == "__main__":
    print(prepare())
