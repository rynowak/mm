"""Baseline the SD3.5 full-FT DINOv2 score against every prior bufo attempt.

Read-only. Embeds two reference sets (canonical curated bufos + teacher cells) once,
then scores each candidate model's existing eval images by per-image max cosine sim to
each reference set. Prints a table so the 0.826 has context. No Claude judgment — just
the same objective metric applied uniformly so we can see if it tracks the user's eye.
"""

from __future__ import annotations

import json
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

REF_SETS = {
    "canon": "/mnt/ray/bufo-data/images",  # user-curated canonical bufos
    "teacher": "/mnt/ray/bufo-data-teacher-v6/images",  # Gemini teacher cells
}
CANDIDATES = {
    "sd35-medium-ft (NEW)": "/mnt/ray/bufo-runs/sd35-medium-ft/eval-1000/images",
    "flux-v6": "/mnt/ray/bufo-runs/flux-soul-v6/eval-500/images",
    "flux-v5": "/mnt/ray/bufo-runs/flux-soul-v5/eval-800/images",
    "flux-v4": "/mnt/ray/bufo-runs/flux-soul-v4/eval-800/images",
    "flux-style": "/mnt/ray/bufo-runs/flux-soul-style/eval-800/images",
    "sdxl-soul": "/mnt/ray/bufo-runs/sdxl-soul/eval-2000-s080-clean/images",
}
EXTS = (".png", ".jpg", ".jpeg")


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

    def embed_dir(d: str) -> torch.Tensor | None:
        if not os.path.isdir(d):
            return None
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(EXTS)]
        if not files:
            return None
        embs = []
        for p in files:
            im = Image.open(p).convert("RGB")
            inp = proc(images=im, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = dino(**inp).last_hidden_state[:, 0]
            embs.append(torch.nn.functional.normalize(out, dim=-1)[0])
        return torch.stack(embs)

    refs = {name: embed_dir(path) for name, path in REF_SETS.items()}
    for name, e in refs.items():
        print(f"ref[{name}] = {0 if e is None else e.shape[0]} images")

    rows = []
    for cand, path in CANDIDATES.items():
        ce = embed_dir(path)
        if ce is None:
            print(f"  SKIP {cand}: no images at {path}")
            continue
        row: dict = {"model": cand, "n": ce.shape[0]}
        for rname, re_ in refs.items():
            if re_ is None:
                continue
            sim = ce @ re_.T  # [G, R]
            maxs = sim.max(dim=1).values  # nearest ref per generated image
            row[f"{rname}_mean"] = round(float(maxs.mean()), 4)
            row[f"{rname}_min"] = round(float(maxs.min()), 4)
        rows.append(row)

    print("\n=== DINOv2 max-sim to reference set (higher = closer to that set) ===")
    hdr = ["model", "n", "canon_mean", "canon_min", "teacher_mean", "teacher_min"]
    print(" | ".join(f"{h:>14}" if h != "model" else f"{h:<22}" for h in hdr))
    for r in rows:
        print(" | ".join((f"{r.get('model', ''):<22}" if h == "model" else f"{str(r.get(h, '-')):>14}") for h in hdr))
    with open("/mnt/ray/bufo-keep/baseline_scores.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote /mnt/ray/bufo-keep/baseline_scores.json")


if __name__ == "__main__":
    main()
