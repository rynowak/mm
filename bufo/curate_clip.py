"""Curate the bufo corpus by CLIP-image similarity to hand-picked canonical references.

Qwen's trait-judging was too noisy (it kept an Android logo as "canonical"), so instead
we anchor on a human-chosen reference set (the platonic bufo: round green body, muted
olive, big expressive eyes) and rank every training image by mean cosine similarity to
those references. Writes the ranked list + a "gradient" grid sampled across the ranking,
so a human can see where quality falls off and pick the keep cutoff.

    python -m bufo.curate_clip --data-dir /mnt/ray/bufo-data --refs bufo-a.png,bufo-b.png,...
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import torch
from mm_training import get_device
from PIL import Image, ImageDraw, ImageFont

from bufo.clip_metrics import ClipEmbedder, load_or_build_train_embeddings

CLIP_MODEL = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def main() -> None:
    ap = argparse.ArgumentParser(description="CLIP-reference curation ranking")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--refs", required=True, help="comma-separated reference image file names")
    ap.add_argument("--out-name", default="curation_clip_rank.jsonl")
    ap.add_argument("--grid-out", default="/mnt/ray/bufo-data/_clip_gradient.png")
    args = ap.parse_args()

    device = get_device()
    embedder = ClipEmbedder.load(CLIP_MODEL, device)
    train_emb, names = load_or_build_train_embeddings(embedder, args.data_dir)  # [N,D] L2-normalized (cached)
    name_idx = {n: i for i, n in enumerate(names)}

    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    ref_idx = [name_idx[r] for r in refs if r in name_idx]
    missing = [r for r in refs if r not in name_idx]
    if missing:
        print(f"WARN missing refs (skipped): {missing}", flush=True)
    ref_embs = train_emb[ref_idx]  # [R,D]
    score = (train_emb @ ref_embs.T).mean(dim=1)  # mean cosine sim to the reference set

    order = torch.argsort(score, descending=True).tolist()
    ranked = [(names[i], float(score[i])) for i in order]
    n = len(ranked)

    root = Path(args.data_dir)
    (root / args.out_name).write_text(
        "\n".join(
            json.dumps({"file_name": fn, "score": round(s, 4), "rank": r + 1}) for r, (fn, s) in enumerate(ranked)
        )
        + "\n"
    )
    print(f"RANKED {n} (refs used {len(ref_idx)}/{len(refs)}) -> {root / args.out_name}", flush=True)
    for q in (50, 100, 200, 300, 400, 500, 700, 1000, n):
        if q <= n:
            print(f"  rank {q:>4}: score {ranked[q - 1][1]:.3f}", flush=True)

    # Gradient grid: 48 images sampled evenly across the full ranking, labeled rank+score.
    images = root / "images"
    cell, cols, pick = 200, 6, 48
    idxs = sorted({round(i * (n - 1) / (pick - 1)) for i in range(pick)})
    rows = math.ceil(len(idxs) / cols)
    g = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    d = ImageDraw.Draw(g)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    for j, oi in enumerate(idxs):
        fn, s = ranked[oi]
        r, c = divmod(j, cols)
        x, y = c * cell, r * cell
        g.paste(Image.open(images / fn).convert("RGB").resize((cell, cell)), (x, y))
        d.rectangle([x, y, x + 96, y + 24], fill=(0, 0, 0))
        d.text((x + 3, y + 2), f"#{oi + 1} {s:.2f}", fill=(255, 255, 0), font=font)
    g.save(args.grid_out)
    print("GRIDB64_BEGIN")
    print(base64.b64encode(Path(args.grid_out).read_bytes()).decode())
    print("GRIDB64_END")


if __name__ == "__main__":
    main()
