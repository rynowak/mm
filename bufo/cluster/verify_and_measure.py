"""Verify the HF upload is intact, then measure NFS reclaim candidates. Read-only."""

from __future__ import annotations

import os
import subprocess

from huggingface_hub import HfApi


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    token = os.environ["HF_WRITE_TOKEN"]
    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/bufo-sd35-medium-ft"
    print("=== HF repo files (size MB) ===")
    info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    for s in info.siblings or []:
        sz = (s.size or 0) / 1e6
        print(f"  {s.rfilename:50s} {sz:10.1f} MB")

    print("\n=== NFS run dir breakdown ===")
    print(sh("du -sh /mnt/ray/bufo-runs/sd35-medium-ft/* 2>/dev/null"))
    print("--- checkpoint-1000 contents ---")
    ckpt = "/mnt/ray/bufo-runs/sd35-medium-ft/checkpoint-1000"
    print(sh(f"du -sh {ckpt}/* 2>/dev/null; ls -la {ckpt}/ 2>/dev/null"))
    print("--- keep dir ---")
    print(sh("du -sh /mnt/ray/bufo-keep/* 2>/dev/null"))
    print("--- staging dirs (deletable) ---")
    print(sh("du -sh /mnt/ray/sd35-stage-* 2>/dev/null"))
    print("=== NFS free ===")
    print(sh("df -h /mnt/ray | tail -1"))


if __name__ == "__main__":
    main()
