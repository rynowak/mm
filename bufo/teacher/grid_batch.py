"""Scale the grid->grid recipe: many themed grids -> a diverse bufo pool.

Reuses the approved input grid as the identity anchor. Each output grid requests 9 cells of
the same character doing different things, rotating through a ~75-item diversity matrix
(poses, gestures, expressions, props) so coverage is broad and each grid is distinct.
Resumable: existing grids are skipped. Slice with slice_grids.py afterward.

Run locally: needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import importlib.util
import os
import traceback

_spec = importlib.util.spec_from_file_location("oe", os.path.join(os.path.dirname(__file__), "openrouter_edit.py"))
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

INPUT_GRID = os.path.expanduser("~/Bufo/grid/input_grid.png")
OUTDIR = os.path.expanduser(os.environ.get("OUTDIR", "~/Bufo/dataset2/grids"))
MODEL = "google/gemini-2.5-flash-image"
N_GRIDS = int(os.environ.get("N_GRIDS", "60"))
# Fewer cells per grid => each bufo is rendered larger (sharper training data). 2x2 -> ~512px
# cells vs 3x3 -> ~300px (the blur cause).
COLS = int(os.environ.get("COLS", "2"))
ROWS = int(os.environ.get("ROWS", "2"))
CELLS = COLS * ROWS

POSES = [
    "standing upright",
    "walking mid-stride",
    "running",
    "jumping with both arms up",
    "sitting cross-legged",
    "lying on its belly propped on its elbows",
    "kneeling on one knee",
    "crouching low",
    "dancing with one arm raised",
    "stretching both arms up overhead",
    "leaning casually to one side",
    "tip-toeing",
    "sitting hugging its knees",
    "lying on its back",
    "marching",
    "balancing on one foot",
    "bowing forward",
    "seen from behind with its back turned",
]
GESTURES = [
    "waving one hand",
    "giving a thumbs up",
    "giving a thumbs down",
    "pointing forward",
    "clapping both hands",
    "shrugging with both palms up",
    "with arms crossed",
    "with hands on its hips",
    "saluting",
    "covering its face with one hand",
    "praying with both hands together",
    "making an ok sign",
    "flexing one arm",
    "blowing a kiss",
    "making a peace sign",
    "raising one fist",
    "beckoning with one hand",
    "scratching its head",
]
EXPRESSIONS = [
    "smiling happily",
    "laughing with eyes squeezed shut",
    "frowning sadly",
    "crying with big tears",
    "angry with a furrowed brow",
    "surprised with wide eyes",
    "with a smug smirk",
    "sleepy with half-closed eyes",
    "with heart-shaped eyes in love",
    "scared with wide eyes",
    "blushing with a shy smile",
    "winking one eye",
    "with a confused puzzled look",
    "with a bored flat expression",
    "with an amazed mind-blown look",
    "with a cheerful grin",
    "with a worried nervous look",
    "with a determined look",
    "sulking with a pout",
    "yawning",
]
PROPS = [
    "holding a coffee mug",
    "holding a red heart",
    "holding a slice of pizza",
    "reading an open book",
    "holding a balloon",
    "holding a bouquet of flowers",
    "holding a game controller",
    "wearing headphones with music notes",
    "looking at a phone",
    "holding a wrapped gift",
    "holding a slice of cake",
    "holding a lit torch",
    "holding a bucket of popcorn",
    "wearing a wizard hat",
    "wearing a crown",
    "wearing sunglasses",
    "wearing a party hat",
    "wearing a chef hat",
    "wearing a cowboy hat",
]
MASTER = POSES + GESTURES + EXPRESSIONS + PROPS

TEMPLATE = (
    f"This image is a grid of the same bufo frog character (a flat green cartoon frog). Create a new image "
    f"that is a {COLS}x{ROWS} grid ({COLS} columns, {ROWS} rows) of this exact same character, each cell "
    f"showing it differently: {{items}}. Make each cell large, sharp, and detailed, filling its cell. Keep "
    f"the character's identity, colors, and flat soft-shaded cartoon style identical in every cell. Plain "
    f"white background with even spacing between cells."
)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"matrix items: {len(MASTER)} | grids: {N_GRIDS} x {CELLS} cells = {N_GRIDS * CELLS} target")
    for i in range(N_GRIDS):
        out = os.path.join(OUTDIR, f"grid_{i:02d}.png")
        if os.path.exists(out):
            print(f"[{i + 1}/{N_GRIDS}] skip grid_{i:02d}")
            continue
        sel = [MASTER[(i * CELLS + j) % len(MASTER)] for j in range(CELLS)]
        prompt = TEMPLATE.format(items="; ".join(sel))
        try:
            oe.edit_image(INPUT_GRID, prompt, MODEL, out)
            print(f"[{i + 1}/{N_GRIDS}] ok   grid_{i:02d}  ({sel[0]}, {sel[1]}, ...)")
        except Exception as e:  # noqa: BLE001 - one grid failing shouldn't abort the batch
            print(f"[{i + 1}/{N_GRIDS}] FAIL grid_{i:02d}: {e}")
            traceback.print_exc()
    print("DONE grids in", OUTDIR)


if __name__ == "__main__":
    main()
