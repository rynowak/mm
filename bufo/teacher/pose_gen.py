"""Two-pass pose engine: generate a pose from the reference sheet, then identity-correct it.

For each target pose: pass-1 generates the character in that pose (gets the pose, may drift),
pass-2 is a close edit conditioned on the real refs that keeps the pose but fixes identity.
Shows a grid of the corrected poses so we can judge pose BREADTH. All-positive phrasing.

Run locally: needs OPENROUTER_API_KEY. Resumable (existing pass-2 outputs skipped).
"""

from __future__ import annotations

import importlib.util
import os
import traceback

from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location("oe", os.path.join(os.path.dirname(__file__), "openrouter_edit.py"))
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

REFS_DIR = os.path.expanduser("~/Bufo/seeds/refs")
OUTDIR = os.path.expanduser("~/Bufo/poses")
GEN_MODEL = "google/gemini-2.5-flash-image"
FIX_MODEL = "google/gemini-2.5-flash-image"
REFS = sorted(os.path.join(REFS_DIR, f) for f in os.listdir(REFS_DIR) if f.lower().endswith(".png"))

PREAMBLE = "The attached images all show the same bufo frog character from different angles and poses."
IDENTITY = (
    "this exact bufo frog character - a simple flat cartoon frog with a wide flat-topped head, "
    "two round eyes resting flat on the top of the head, a small smooth mouth, a flat slim body, "
    "and simple rounded hands, drawn in the exact same colors as the reference images"
)
CONSTRAINTS = (
    "Match the reference exactly: a flat-topped head with flat-resting eyes, a flat slim body, a smooth "
    "mouth, the exact same colors, and a flat soft-shaded cartoon sticker style with simple flat shapes. "
    "Keep the hands empty. Plain white background."
)
FIX_PROMPT = (
    "The first attached image shows the bufo frog character in a pose. The remaining images are references "
    "of the correct bufo character. Redraw the first image keeping its exact same pose, composition, and "
    "gesture, but make the character's head shape, body shape, proportions, and colors match the reference "
    "bufo exactly: a wide flat-topped head with round eyes resting flat on top, a flat slim body, a smooth "
    "mouth, the exact reference colors, and a flat soft-shaded cartoon sticker style. Plain white background."
)

POSES = {
    "walk": "walking, mid-stride with one leg forward, full body",
    "run": "running energetically mid-stride, full body",
    "jump": "jumping into the air with both arms raised, full body",
    "sit_cross": "sitting cross-legged on the ground, full body",
    "lie_belly": "lying on its belly propped up on both elbows, full body",
    "arms_crossed": "standing with both arms crossed, full body",
    "hands_hips": "standing with both hands on its hips, full body",
    "shrug": "standing and shrugging with both hands raised palms-up, full body",
    "dance": "dancing happily with one arm up and one leg lifted, full body",
    "kneel": "kneeling on one knee, full body",
    "back_view": "seen from behind with its back turned to the viewer, full body",
    "point": "standing and pointing forward with one hand, full body",
}


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    for i, (key, pose) in enumerate(POSES.items()):
        p1 = os.path.join(OUTDIR, f"{key}__pass1.png")
        p2 = os.path.join(OUTDIR, f"{key}__pass2.png")
        if os.path.exists(p2):
            print(f"[{i + 1}/{len(POSES)}] skip {key}")
            continue
        try:
            gen = f"{PREAMBLE} Draw {IDENTITY}, now {pose}. {CONSTRAINTS}"
            oe.edit_image(REFS, gen, GEN_MODEL, p1)
            oe.edit_image([p1, *REFS], FIX_PROMPT, FIX_MODEL, p2)
            print(f"[{i + 1}/{len(POSES)}] ok   {key}")
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(POSES)}] FAIL {key}: {e}")
            traceback.print_exc()

    keys = list(POSES.keys())
    cell, cols = 240, 4
    rows = (len(keys) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, key in enumerate(keys):
        p = os.path.join(OUTDIR, f"{key}__pass2.png")
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 10, cell - 10))
            sheet.paste(im, (x + 5, y + 5))
        d.text((x + 4, y + cell), key, fill=(0, 0, 0))
    grid = os.path.join(OUTDIR, "POSES.jpg")
    sheet.save(grid, quality=90)
    print("wrote", grid)


if __name__ == "__main__":
    main()
