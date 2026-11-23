#!/usr/bin/env python3
"""Candidate configurations. Only live probe/quality evidence can validate a recipe."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("balanced", "max-context", "ngram", "dspark", "dflash", "eagle", "eagle-native")


def effective_argument(cmd, flag):
    indices = [i for i, value in enumerate(cmd) if value == flag]
    return cmd[indices[-1] + 1] if indices else None


def validate_checkpoints(cmd):
    for flag in ("--model-path", "--speculative-draft-model-path"):
        value = effective_argument(cmd, flag)
        if value is None:
            continue
        path = Path(value)
        if not (path / "config.json").exists():
            raise ValueError(f"Missing local checkpoint: {path}; run scripts/download.sh first")
        if (path / ".conversion-in-progress").exists():
            raise ValueError(f"Incomplete checkpoint conversion: {path}")
        manifest = path / "revision.json"
        if manifest.exists() and not json.loads(manifest.read_text()).get("complete"):
            raise ValueError(f"Incomplete checkpoint download: {path}")


def command(profile, extra=()):
    target = os.environ.get("MODEL_PATH", str(ROOT / "weights/target"))
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", target,
           "--trust-remote-code", "--served-model-name", "qwen3.8-27b",
           "--kv-cache-dtype", "fp8_e4m3", "--mem-fraction-static", "0.88",
           "--attention-backend", "flashinfer", "--max-running-requests", "1",
           "--cuda-graph-max-bs-decode", "1", "--reasoning-parser", "qwen3",
           "--tool-call-parser", "qwen3_coder", "--language-model-only",
           "--mamba-ssm-dtype", "bfloat16", "--mamba-radix-cache-strategy",
           "extra_buffer_lazy", "--chunked-prefill-size", "2048",
           "--host", os.environ.get("HOST", "127.0.0.1"),
           "--port", os.environ.get("PORT", "30000")]
    if profile == "max-context":
        cmd += ["--disable-radix-cache"]
    elif profile == "ngram":
        cmd += ["--speculative-algorithm", "NGRAM"]
    elif profile == "dspark":
        cmd += ["--speculative-algorithm", "DSPARK", "--speculative-draft-model-path",
                os.environ.get("DRAFT_MODEL_PATH", str(ROOT / "weights/dspark")),
                "--speculative-draft-model-quantization", "modelopt_fp4",
                "--speculative-draft-attention-backend", "flashinfer",
                "--max-mamba-cache-size", "8"]
    elif profile == "dflash":
        # Deliberately refers to an offline-quantized candidate, not the BF16 original.
        cmd += ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path",
                os.environ.get("DRAFT_MODEL_PATH", str(ROOT / "weights/dflash-nvfp4")),
                "--speculative-num-draft-tokens", "8"]
    elif profile in ("eagle", "eagle-native"):
        draft = os.environ.get("DRAFT_MODEL_PATH") or effective_argument(
            list(extra), "--speculative-draft-model-path")
        if "MODEL_PATH" not in os.environ and not draft:
            raise ValueError("EAGLE requires MODEL_PATH with an MTP head or an external DRAFT_MODEL_PATH")
        cmd += ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3",
                "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
                "--speculative-draft-model-path", draft or target]
        if profile == "eagle":
            cmd += ["--enable-linear-replayssm-spec"]
    # Later scalar arguments override earlier ones in argparse; boolean removal
    # (e.g. vision) is explicit, never a second contradictory boolean argument.
    extra = list(extra)
    if "--with-vision" in extra:
        extra.remove("--with-vision")
        cmd.remove("--language-model-only")
    return cmd + extra


def main():
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("MAX_JOBS", "2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    cmd = command(args.profile, extra)
    if args.dry_run:
        print(shlex.join(cmd))
        return
    validate_checkpoints(cmd)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
