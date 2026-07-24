"""Two-pass pose pipeline test: identity-correct a drifted pose output.

Pass 1 (already done by bakeoff) gets the POSE right but drifts identity. Pass 2 is a close
edit on that output, conditioned on the real bufo references, told to keep the exact pose but
match the character's head/body/colors to the reference. If this pulls drifted poses back
on-model, it's our pose engine. Builds a before/after montage.

Run locally: needs OPENROUTER_API_KEY. All-positive phrasing.
"""

from __future__ import annotations

import importlib.util
import os
import traceback

from PIL import Image, ImageDraw

_spec = importlib.util.spec_from_file_location("oe", os.path.join(os.path.dirname(__file__), "openrouter_edit.py"))
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

V3 = os.path.expanduser("~/Bufo/bakeoff-v3")
REFS_DIR = os.path.expanduser("~/Bufo/seeds/refs")
OUTDIR = os.path.expanduser("~/Bufo/twopass")
CORRECTOR = "google/gemini-2.5-flash-image"  # the most identity-faithful model from the bake-off

# drifted pose outputs to correct (good pose, off identity)
STEP1 = ["stand_point__NB2", "laugh__NBpro", "sit_wave__NB2", "angle34__NBpro"]
REFS = sorted(os.path.join(REFS_DIR, f) for f in os.listdir(REFS_DIR) if f.lower().endswith(".png"))

PROMPT = (
    "The first attached image shows the bufo frog character in a pose. The remaining images are "
    "references of the correct bufo character. Redraw the first image keeping its exact same pose, "
    "composition, and gesture, but make the character's head shape, body shape, proportions, and colors "
    "match the reference bufo exactly: a wide flat-topped head with round eyes resting flat on top, a "
    "flat slim body, a smooth mouth, the exact reference colors, and a flat soft-shaded cartoon sticker "
    "style with simple flat shapes. Plain white background."
)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    done = []
    for i, name in enumerate(STEP1):
        src = os.path.join(V3, f"{name}.png")
        out = os.path.join(OUTDIR, f"{name}__corrected.png")
        if not os.path.exists(src):
            print(f"[{i + 1}/{len(STEP1)}] missing {src}")
            continue
        try:
            oe.edit_image([src, *REFS], PROMPT, CORRECTOR, out)
            print(f"[{i + 1}/{len(STEP1)}] ok   {os.path.basename(out)}")
            done.append((name, src, out))
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(STEP1)}] FAIL {name}: {e}")
            traceback.print_exc()

    # before/after montage: rows = item, cols = [pass1 drifted, pass2 corrected]
    cell, gut = 300, 110
    sheet = Image.new("RGB", (gut + 2 * cell, 24 + len(STEP1) * cell), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((gut + cell // 2 - 30, 6), "pass1 (drifted)", fill=(0, 0, 0))
    d.text((gut + cell + cell // 2 - 30, 6), "pass2 (corrected)", fill=(0, 0, 0))
    for r, name in enumerate(STEP1):
        y = 24 + r * cell
        d.text((4, y + cell // 2), name, fill=(0, 0, 0))
        for c, p in enumerate([os.path.join(V3, f"{name}.png"), os.path.join(OUTDIR, f"{name}__corrected.png")]):
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 10, cell - 10))
            sheet.paste(im, (gut + c * cell + 5, y + 5))
    grid = os.path.join(OUTDIR, "BEFORE_AFTER.jpg")
    sheet.save(grid, quality=90)
    print("wrote", grid)


if __name__ == "__main__":
    main()
