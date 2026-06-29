"""Secure the SD3.5 full-FT checkpoint on NFS and inventory prior eval outputs.

Run on the cluster. Copies checkpoint-1000/transformer to a protected keep-dir
(insurance against training-run checkpoint rotation clobbering it), stashes the exact
dataset metadata next to the weights, and lists candidate v5/v6 eval image dirs so we
can baseline the DINOv2 number against earlier attempts.
"""

from __future__ import annotations

import os
import shutil
import subprocess

SRC = "/mnt/ray/bufo-runs/sd35-medium-ft/checkpoint-1000/transformer"
KEEP = "/mnt/ray/bufo-keep/sd35-medium-ft-1000"
DATA_META = "/mnt/ray/bufo-data-sd35full/metadata.jsonl"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    print("=== NFS before ===")
    print(sh("df -h /mnt/ray | tail -1"))
    print("=== source ===", SRC, "exists=", os.path.isdir(SRC))
    if os.path.isdir(SRC):
        print(sh(f"du -sh {SRC}"))
        os.makedirs(KEEP, exist_ok=True)
        dst = os.path.join(KEEP, "transformer")
        if os.path.isdir(dst):
            print("keep/transformer already exists, skipping copy")
        else:
            shutil.copytree(SRC, dst)
            print("copied ->", dst)
        if os.path.exists(DATA_META):
            shutil.copy(DATA_META, os.path.join(KEEP, "dataset_metadata.jsonl"))
        print(sh(f"du -sh {KEEP}; ls -la {KEEP}; ls -la {dst}"))
    print("=== NFS after ===")
    print(sh("df -h /mnt/ray | tail -1"))

    print("\n=== candidate prior-eval image dirs on NFS (for baseline) ===")
    print(sh("ls -d /mnt/ray/bufo-runs/*/ 2>/dev/null"))
    print("--- dirs containing generated images ---")
    print(sh("find /mnt/ray/bufo-runs -maxdepth 3 -type d -name images 2>/dev/null | head -40"))
    print("--- flux/v5/v6 hints ---")
    print(sh("ls -d /mnt/ray/*flux* /mnt/ray/*v5* /mnt/ray/*v6* 2>/dev/null"))


if __name__ == "__main__":
    main()
