"""Dataset review heuristics for the bufo corpus — fast, offline, no model.

Surfaces the signals you need to decide what to keep before a retrain: caption
length / scene-iness, aspect, transparency coverage, near-duplicate groups, and a
cheap text-heavy proxy. Emits ``curate-report.{json,csv}`` and a printed summary.
Model-based quality signals (CLIP style, VLM flags) come later, in recaption.

    uv run python -m bufo.curate report
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from bufo.data import filename_phrase

SCENE_TOKENS = 6  # >= this many action words -> "scene-like" narrative name
TEXT_TOKENS = 8  # >= this many -> likely text-heavy / a caption-in-image
DUP_HAMMING = 5  # average-hash distance below which two images are near-dups


@dataclass
class ImageStats:
    file_name: str
    n_tokens: int
    scene_like: bool
    text_proxy: bool
    width: int
    height: int
    aspect: float
    alpha_coverage: float
    ahash: str
    dup_group: int  # -1 = unique, else shared group id


def average_hash(img: Image.Image, size: int = 8) -> int:
    """8x8 average hash (hand-rolled — no ``imagehash`` dependency)."""
    gray = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    px = cast("list[int]", list(gray.getdata()))
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")  # int.bit_count() is py3.11+; this is py3.9-safe (cluster runs 3.9)


def alpha_coverage(img: Image.Image) -> float:
    """Fraction of (downsampled) pixels that are opaque — a subject-size proxy."""
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode != "RGBA":
        return 1.0
    a = img.getchannel("A").resize((64, 64), Image.Resampling.LANCZOS)
    px = cast("list[int]", list(a.getdata()))
    return sum(1 for p in px if p > 16) / len(px)


def analyze_image(path: Path) -> ImageStats:
    n_tokens = len(filename_phrase(path.name).split())
    with Image.open(path) as im:
        w, h = im.size
        return ImageStats(
            file_name=path.name,
            n_tokens=n_tokens,
            scene_like=n_tokens >= SCENE_TOKENS,
            text_proxy=n_tokens >= TEXT_TOKENS,
            width=w,
            height=h,
            aspect=round(w / h, 3) if h else 0.0,
            alpha_coverage=round(alpha_coverage(im), 3),
            ahash=f"{average_hash(im):016x}",
            dup_group=-1,
        )


def assign_dup_groups(stats: list[ImageStats], threshold: int = DUP_HAMMING) -> None:
    """Union near-duplicate images (by average-hash Hamming distance) in place."""
    hashes = [int(s.ahash, 16) for s in stats]
    parent = list(range(len(stats)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            if _hamming(hashes[i], hashes[j]) <= threshold:
                parent[find(i)] = find(j)
    # Only assign group ids to images that share with at least one other.
    sizes: dict[int, int] = {}
    for i in range(len(stats)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    gid: dict[int, int] = {}
    for i, s in enumerate(stats):
        root = find(i)
        if sizes[root] > 1:
            s.dup_group = gid.setdefault(root, len(gid))


def report(data_dir: str | Path = "bufo/data", *, dup_threshold: int = DUP_HAMMING) -> dict:
    """Analyze the corpus, write report files, return + print a summary."""
    root = Path(data_dir)
    img_dir = root / "raw" if (root / "raw").is_dir() else root / "images"
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs in {img_dir} — run `python -m bufo.prepare` first.")
    print(f"Analyzing {len(paths)} images in {img_dir}...")
    stats = [analyze_image(p) for p in paths]
    assign_dup_groups(stats, dup_threshold)

    n = len(stats)
    n_groups = len({s.dup_group for s in stats if s.dup_group >= 0})
    n_in_dups = sum(1 for s in stats if s.dup_group >= 0)
    summary = {
        "total": n,
        "scene_like": sum(s.scene_like for s in stats),
        "text_proxy": sum(s.text_proxy for s in stats),
        "low_alpha_lt_0_15": sum(s.alpha_coverage < 0.15 for s in stats),
        "non_square": sum(abs(s.aspect - 1.0) > 0.1 for s in stats),
        "near_dup_groups": n_groups,
        "images_in_dup_groups": n_in_dups,
        "median_tokens": sorted(s.n_tokens for s in stats)[n // 2],
    }

    report_blob = {"summary": summary, "images": [asdict(s) for s in stats]}
    (root / "curate-report.json").write_text(json.dumps(report_blob, indent=2))
    with open(root / "curate-report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0])))
        writer.writeheader()
        writer.writerows(asdict(s) for s in stats)

    print(f"  total                {summary['total']}")
    print(f"  scene-like (>={SCENE_TOKENS} words)  {summary['scene_like']}")
    print(f"  text-heavy proxy     {summary['text_proxy']}")
    print(f"  low alpha (<0.15)    {summary['low_alpha_lt_0_15']}")
    print(f"  non-square           {summary['non_square']}")
    print(f"  near-dup groups      {summary['near_dup_groups']} ({summary['images_in_dup_groups']} images)")
    print(f"  median action words  {summary['median_tokens']}")
    print(f"Wrote curate-report.json + .csv to {root}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Review the bufo dataset")
    parser.add_argument("command", choices=["report"], help="what to do")
    parser.add_argument("--data-dir", type=str, default="bufo/data")
    parser.add_argument("--dup-threshold", type=int, default=DUP_HAMMING)
    args = parser.parse_args()
    if args.command == "report":
        report(args.data_dir, dup_threshold=args.dup_threshold)


if __name__ == "__main__":
    main()
