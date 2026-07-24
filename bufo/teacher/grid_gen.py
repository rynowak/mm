"""Grid -> grid bufo generation (the recipe that works manually).

Compose a grid of clean bufo samples into ONE input image, then ask the model for a new
grid of the same character in varied poses/expressions. A single coherent image edit keeps
the character consistent across cells (unlike separate-image generation, which drifts).

Run locally: needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import importlib.util
import math
import os

from PIL import Image

_spec = importlib.util.spec_from_file_location("oe", os.path.join(os.path.dirname(__file__), "openrouter_edit.py"))
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

OUTDIR = os.path.expanduser("~/Bufo/grid")
MODEL = "google/gemini-2.5-flash-image"

SEEDS = os.path.expanduser("~/Bufo/seeds")
REFS = os.path.expanduser("~/Bufo/seeds/refs")
INPUTS = [os.path.join(REFS, f) for f in sorted(os.listdir(REFS)) if f.endswith(".png")] + [
    os.path.join(SEEDS, f"seed{n}.png") for n in (1, 2, 3, 4, 6)
]

PROMPT = (
    "This image is a grid of the same bufo frog character (a flat green cartoon frog). Create a new image "
    "that is a 3x4 grid (3 columns, 4 rows) of this exact same character, each cell showing it in a "
    "different pose or expression: standing, sitting, walking, jumping, waving, thumbs up, arms crossed, "
    "pointing, laughing, crying, surprised, and angry. Keep the character's identity, colors, and flat "
    "soft-shaded cartoon style identical in every cell. Plain white background with even spacing between cells."
)


def build_input_grid(paths: list[str], cell: int = 400, cols: int = 3) -> str:
    rows = math.ceil(len(paths) / cols)
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGBA")
        im.thumbnail((cell - 24, cell - 24))
        x = (i % cols) * cell + (cell - im.width) // 2
        y = (i // cols) * cell + (cell - im.height) // 2
        grid.paste(im, (x, y), im)
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "input_grid.png")
    grid.save(out)
    return out


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    gin = build_input_grid(INPUTS)
    print("input grid:", gin, "from", len(INPUTS), "bufos")
    gout = os.path.join(OUTDIR, "output_grid.png")
    oe.edit_image(gin, PROMPT, MODEL, gout)
    print("output grid:", gout)


if __name__ == "__main__":
    main()
