#!/usr/bin/env python3
"""Execute the SM120 CUTLASS FP4 quantize/GEMM path on a small known workload."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fp4-smoke.json"))
    args = parser.parse_args()
    import torch
    import flashinfer
    result = {"at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "backend": "cutlass", "shape_mnk": [128, 128, 512]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result["device"] = torch.cuda.get_device_name()
        result["capability"] = torch.cuda.get_device_capability()
        torch.manual_seed(1701)
        a = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        ga = (448 * 6 / a.float().abs().max()).reshape(1)
        gb = (448 * 6 / b.float().abs().max()).reshape(1)
        start = time.monotonic()
        aq, sa = flashinfer.fp4_quantize(a, ga)
        bq, sb = flashinfer.fp4_quantize(b, gb)
        out = flashinfer.mm_fp4(aq, bq.T, sa, sb.T, 1 / (ga * gb), backend="cutlass")
        torch.cuda.synchronize()
        ref = a.float() @ b.float().T
        error = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt().item()
        result.update(first_call_seconds=time.monotonic() - start, relative_rmse=error,
                      finite=bool(torch.isfinite(out).all()), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if not result["finite"] or error > .2:
            raise RuntimeError(f"FP4 GEMM numerical check failed: relative RMSE={error}")
        result["status"] = "passed"
    except Exception as e:
        result.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
