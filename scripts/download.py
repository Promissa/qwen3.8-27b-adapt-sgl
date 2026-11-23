#!/usr/bin/env python3
"""Pin a Hub revision on ext4 and reuse its completed files across retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

MODELS = {
    "target": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
    "reference": "RadixArk/Qwen3.8-27B-NVFP4",
    "dspark": "gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4",
    "dflash": "incoai/Qwen3.8-27B-DFlash2",
    "qat": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS, nargs="?", default="target")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    root = Path(__file__).resolve().parents[1]
    dest = root / "weights" / args.model
    dest.mkdir(parents=True, exist_ok=True)
    fs = subprocess.check_output(["df", "-T", str(dest)], text=True)
    if str(dest).startswith("/mnt/") or " drvfs " in fs or " 9p " in fs:
        raise SystemExit("Place weights on the WSL ext4 filesystem, not a Windows mount")
    pin = dest / "revision.json"
    if pin.exists():
        manifest = json.loads(pin.read_text())
        if manifest["repo_id"] != MODELS[args.model]:
            raise SystemExit("Existing download belongs to a different model")
        revision = manifest["revision"]
        if args.revision not in ("main", revision):
            raise SystemExit("Use a new destination for a different revision")
    else:
        revision = HfApi().model_info(MODELS[args.model], revision=args.revision).sha
        manifest = {"repo_id": MODELS[args.model], "revision": revision,
                    "started_at": datetime.now(timezone.utc).isoformat(), "complete": False}
        pin.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot_download(MODELS[args.model], revision=revision, local_dir=dest,
                      ignore_patterns=["*.gguf", "*.bin", "*.pt", "*.pth"])
    manifest["complete"] = True
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    pin.write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
