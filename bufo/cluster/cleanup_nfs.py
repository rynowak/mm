"""Reclaim NFS after the SD3.5 checkpoint is durably on HF.

Deletes only known-redundant paths in our own run dir + staging:
  - checkpoint-1000 (transformer mirrored in keep-dir + HF; optimizer state not needed
    for inference — a fresh run is the clean way to add steps)
  - text_encoder*/vae copies (identical to the base model)
  - the staging dir
Keeps: eval-1000 (contact sheet + images), logs, and the keep-dir transformer.
"""

from __future__ import annotations

import shutil
import subprocess

RUN = "/mnt/ray/bufo-runs/sd35-medium-ft"
DELETE = [
    f"{RUN}/checkpoint-1000",
    f"{RUN}/text_encoder",
    f"{RUN}/text_encoder_2",
    f"{RUN}/text_encoder_3",
    f"{RUN}/vae",
    f"{RUN}/tokenizer",
    f"{RUN}/tokenizer_2",
    f"{RUN}/tokenizer_3",
    f"{RUN}/scheduler",
    f"{RUN}/model_index.json",
    "/mnt/ray/sd35-stage-sd35-medium-ft",
]


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    print("=== before ===", sh("df -h /mnt/ray | tail -1"))
    for p in DELETE:
        try:
            shutil.rmtree(p)
            print("rmtree", p)
        except NotADirectoryError:
            import os

            os.remove(p)
            print("rm", p)
        except FileNotFoundError:
            print("absent (ok)", p)
    print("=== kept in run dir ===", sh(f"ls -la {RUN}"))
    print("=== keep dir intact ===", sh("du -sh /mnt/ray/bufo-keep/sd35-medium-ft-1000"))
    print("=== after ===", sh("df -h /mnt/ray | tail -1"))


if __name__ == "__main__":
    main()
