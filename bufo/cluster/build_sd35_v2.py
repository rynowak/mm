"""Assemble the v2 SD3.5 training set: curated canon_v2 + teacher grid-edit pose cells.

canon_v2 (real curated bufos) anchors identity; the grid-edit cells add pose/expression
diversity. canon_v2 captions are normalized to the winning schema; the pose cells are already
in-schema. Output: an imagefolder at OUT (images + metadata.jsonl).
"""

from __future__ import annotations

import json
import os
import shutil

OUT = os.environ.get("SD35_OUT", "/mnt/ray/bufo-data-sd35-v2")
PREFIX = "olive green adult bufo"
SUFFIX = "soft-shaded cartoon sticker"
# (metadata.jsonl, image_root, normalize_caption). Second source (teacher data) is env-configurable.
SRCS = [
    ("bufo/data_canon_v2/metadata.jsonl", "/mnt/ray/bufo-data/images", True),
    (
        os.environ.get("EXTRA_META", "/mnt/ray/bufo-data-poses/metadata.jsonl"),
        os.environ.get("EXTRA_IMG", "/mnt/ray/bufo-data-poses/images"),
        False,
    ),
]


def normalize(caption: str) -> str:
    parts = [p.strip() for p in caption.split(",")]
    while parts and parts[0].lower() in {"bufo", "green bufo", "olive green bufo", "green", "olive", PREFIX}:
        parts.pop(0)
    drop = {"flat cartoon sticker", "soft-shaded cartoon sticker", "big forward-set eyes", "short stubby limbs"}
    parts = [p for p in parts if p.lower() not in drop]
    middle = ", ".join(p for p in parts if p)
    return f"{PREFIX}, {middle}, {SUFFIX}" if middle else f"{PREFIX}, {SUFFIX}"


def main() -> None:
    img_dir = os.path.join(OUT, "images")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(img_dir)
    rows: list[dict] = []
    for meta_path, image_root, do_norm in SRCS:
        n = 0
        for line in open(meta_path):  # noqa: SIM115 (simple read iteration)
            r = json.loads(line)
            fn = os.path.basename(r["file_name"])
            src = os.path.join(image_root, fn)
            if not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(img_dir, fn))
            cap = normalize(r["caption"]) if do_norm else r["caption"]
            rows.append({"file_name": fn, "caption": cap})
            n += 1
        print(f"  {meta_path}: {n} images")
    with open(os.path.join(OUT, "metadata.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL {len(rows)} images -> {OUT}")
    for r in (rows[0], rows[-1]):
        print("  sample:", r["caption"])


if __name__ == "__main__":
    main()
