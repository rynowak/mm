"""Assemble the combined bufo training set for the SD3.5 full-FT, on the cluster.

Sources (images already staged on /mnt/ray):
  - curated real bufos: bufo/data_canon_v2/metadata.jsonl -> /mnt/ray/bufo-data/images
  - teacher cells:       bufo/data_teacher_v6/metadata.jsonl -> /mnt/ray/bufo-data-teacher-v6/images

Output: an imagefolder at OUT (images + metadata.jsonl), captions normalized to the
adult-anchored schema: "olive green adult bufo, {desc}, soft-shaded cartoon sticker".
"""

from __future__ import annotations

import json
import os
import shutil

OUT = "/mnt/ray/bufo-data-sd35full"
SRCS = [
    ("bufo/data_canon_v2/metadata.jsonl", "/mnt/ray/bufo-data/images"),
    ("bufo/data_teacher_v6/metadata.jsonl", "/mnt/ray/bufo-data-teacher-v6/images"),
]
PREFIX = "olive green adult bufo"
SUFFIX = "soft-shaded cartoon sticker"


def normalize(caption: str) -> str:
    """Strip whatever trigger/style wrapper a source used; keep the middle descriptors."""
    parts = [p.strip() for p in caption.split(",")]
    # drop a leading trigger token ("bufo" / "green bufo" / "olive green bufo")
    while parts and parts[0].lower() in {"bufo", "green bufo", "olive green bufo", "green", "olive"}:
        parts.pop(0)
    # drop a trailing style anchor (flat/soft cartoon sticker, eye/limb cues)
    drop_tail = {"flat cartoon sticker", "soft-shaded cartoon sticker", "big forward-set eyes", "short stubby limbs"}
    parts = [p for p in parts if p.lower() not in drop_tail]
    middle = ", ".join(p for p in parts if p)
    return f"{PREFIX}, {middle}, {SUFFIX}" if middle else f"{PREFIX}, {SUFFIX}"


def main() -> None:
    img_dir = os.path.join(OUT, "images")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(img_dir)
    rows: list[dict] = []
    for meta_path, image_root in SRCS:
        n = 0
        for line in open(meta_path):  # noqa: SIM115 (simple read iteration)
            r = json.loads(line)
            fn = os.path.basename(r["file_name"])
            src = os.path.join(image_root, fn)
            if not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(img_dir, fn))
            rows.append({"file_name": fn, "caption": normalize(r["caption"])})
            n += 1
        print(f"  {meta_path}: {n} images")
    with open(os.path.join(OUT, "metadata.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL {len(rows)} images -> {OUT}")
    for r in rows[:3]:
        print("  sample:", r["caption"])


if __name__ == "__main__":
    main()
