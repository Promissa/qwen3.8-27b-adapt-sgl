"""Resume the pinned public ALE v1 Ubuntu disk and verify all publisher SHA256 digests."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import requests

BASE = ("https://huggingface.co/datasets/agents-last-exam/ale-images-qcow2/resolve/"
        "21290f6d47ae765def3c799281e8cf6e4bc22a22/")
BLOCK = 16 << 20


def fetch_block(item, start, end):
    expected = f"bytes {start}-{end}/{item['size']}"
    for attempt in range(9):
        try:
            with requests.get(
                BASE + item["filename"], params={"bytes": f"{start}-{end}", "n": attempt},
                headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 80),
            ) as response:
                response.raise_for_status()
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("unexpected HTTP Content-Range")
                block = b"".join(response.iter_content(chunk_size=1 << 20))
            if len(block) != end - start + 1:
                raise ValueError("incomplete HTTP response")
            return block
        except (requests.RequestException, ValueError) as exc:
            print(item["filename"], start, type(exc).__name__, "attempt", attempt + 1, flush=True)
            if attempt == 8:
                raise RuntimeError(f"Failed to read {item['filename']} at {start}") from None
            time.sleep(min(20, 2 ** attempt))


def transfer(manifest, directory, workers):
    directory.mkdir(exist_ok=True, parents=True)
    final = directory / manifest["filename"]
    if final.exists():
        with final.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == manifest["sha256"]:
                return
        raise ValueError("Existing final file fails the published SHA256")
    partial = directory / (manifest["filename"] + ".partial")
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
    jobs = []
    offset = 0
    for item in manifest["parts"]:
        jobs.append((item, offset))
        offset += item["size"]
    # Size reservation stays sparse on ext4 and allows SEEK_HOLE on untouched ranges.
    os.ftruncate(fd, offset)

    def part(job):
        item, offset = job
        end = offset + item["size"]
        marker = directory / (Path(item["filename"]).name + ".verified")
        # SEEK_HOLE uses the file-description offset. Open independently for each reader.
        with partial.open("rb", buffering=0) as reader:
            pos = min(os.lseek(reader.fileno(), offset, os.SEEK_HOLE), end)
            pos = max(offset, pos - (pos - offset) % BLOCK) if pos != end else end
            for integrity_attempt in range(2):
                while pos < end:
                    length = min(BLOCK, end - pos)
                    data = fetch_block(item, pos - offset, pos - offset + length - 1)
                    if os.pwrite(fd, data, pos) != length:
                        raise OSError("short disk write")
                    pos += length
                reader.seek(offset)
                digest = hashlib.sha256()
                remaining = item["size"]
                while remaining:
                    data = reader.read(min(4 << 20, remaining))
                    if not data:
                        raise OSError("unexpected EOF in guest disk")
                    digest.update(data)
                    remaining -= len(data)
                if digest.hexdigest() == item["sha256"]:
                    os.fsync(fd)
                    marker.write_text(digest.hexdigest())
                    print("verified", item["filename"], flush=True)
                    return
                pos = offset
            raise ValueError(f"Integrity check failed for {item['filename']}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(part, jobs))
        with partial.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != manifest["sha256"]:
                raise ValueError("full disk SHA256 mismatch")
        os.replace(partial, final)
        print("verified full image", manifest["sha256"], flush=True)
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    transfer(json.loads(args.manifest.read_text()), args.directory, args.workers)
