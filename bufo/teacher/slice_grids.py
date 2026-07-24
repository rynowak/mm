"""Slice generated grids into individual bufo cells (content-aware) + contact sheets.

Per grid: find horizontal content bands (rows of bufos) via row projection, then within each
band find vertical bands (individual bufos) via column projection. Robust to uneven spacing
and to grids that aren't exactly 3x3. Each cell is cropped to its content bbox + margin and
saved on white. Builds chunked contact sheets for curation.

Run locally. No API. Reads ~/Bufo/dataset/grids, writes ~/Bufo/dataset/cells + contact sheets.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

GRIDS = os.path.expanduser(os.environ.get("GRIDS_DIR", "~/Bufo/dataset/grids"))
CELLS = os.path.expanduser(os.environ.get("CELLS_DIR", "~/Bufo/dataset/cells"))


def bands(counts: np.ndarray, thresh: float, min_run: int) -> list[tuple[int, int]]:
    res: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(counts):
        if v >= thresh and start is None:
            start = i
        elif v < thresh and start is not None:
            if i - start >= min_run:
                res.append((start, i))
            start = None
    if start is not None and len(counts) - start >= min_run:
        res.append((start, len(counts)))
    return res


def slice_grid(path: str) -> list[Image.Image]:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    h, w = a.shape[:2]
    nonwhite: np.ndarray = np.asarray((a < 245).any(axis=2))
    out: list[Image.Image] = []
    rowbands = bands(nonwhite.sum(axis=1), thresh=max(3, w * 0.01), min_run=int(h * 0.05))
    for r0, r1 in rowbands:
        strip = nonwhite[r0:r1, :]
        colbands = bands(strip.sum(axis=0), thresh=max(3, (r1 - r0) * 0.02), min_run=int(w * 0.05))
        for c0, c1 in colbands:
            sub = nonwhite[r0:r1, c0:c1]
            ys, xs = np.where(sub)
            if len(xs) < 50:  # ignore tiny specks
                continue
            x0, y0, x1, y1 = c0 + int(xs.min()), r0 + int(ys.min()), c0 + int(xs.max()) + 1, r0 + int(ys.max()) + 1
            m = int(max(x1 - x0, y1 - y0) * 0.08)
            crop = im.crop((max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)))
            out.append(crop)
    return out


def main() -> None:
    os.makedirs(CELLS, exist_ok=True)
    grids = sorted(glob.glob(os.path.join(GRIDS, "grid_*.png")))
    all_cells: list[str] = []
    for gp in grids:
        gi = os.path.splitext(os.path.basename(gp))[0]
        cells = slice_grid(gp)
        for j, c in enumerate(cells):
            p = os.path.join(CELLS, f"{gi}_{j:02d}.png")
            c.save(p)
            all_cells.append(p)
        print(f"{gi}: {len(cells)} cells")
    print(f"TOTAL {len(all_cells)} cells")

    # chunked contact sheets for curation
    per, cols, cell = 64, 8, 150
    for ci in range(0, len(all_cells), per):
        chunk = all_cells[ci : ci + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
        for k, p in enumerate(chunk):
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 6, cell - 6))
            x = (k % cols) * cell + (cell - im.width) // 2
            y = (k // cols) * cell + (cell - im.height) // 2
            sheet.paste(im, (x, y))
        sp = os.path.expanduser(f"~/Bufo/dataset/contact_{ci // per:02d}.jpg")
        sheet.save(sp, quality=85)
        print("wrote", sp)


if __name__ == "__main__":
    main()
