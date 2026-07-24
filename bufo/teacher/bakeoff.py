"""Teacher bake-off: edit bufo with several models using a MULTI-POSE reference sheet.

Feeding one waist-up seed and asking for full body makes the model hallucinate a body; the
fix is a character sheet (standing / sitting / angles / hands) so every requested pose has a
reference. For each edit-prompt x model we feed the whole sheet and produce the character in
a new pose, then build a grid (rows = edit, cols = model) for the USER to pick the teacher.

Run locally: needs OPENROUTER_API_KEY in env. Resumable (existing outputs skipped).
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
OUTDIR = os.path.expanduser("~/Bufo/bakeoff-v3")
REFS = sorted(os.path.join(REFS_DIR, f) for f in os.listdir(REFS_DIR) if f.lower().endswith(".png"))

PREAMBLE = "The attached images all show the same bufo frog character from different angles and poses."
# All-POSITIVE phrasing (these models latch onto nouns inside "no X"). Identity anchored on the
# flat silhouette + "exact same colors as the reference" so the true palette is preserved.
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
_EDITS = {
    "angle34": "turned to a three-quarter side view with a calm friendly closed-mouth smile",
    "laugh": "laughing happily with eyes squeezed shut and a wide open smooth-lipped smile, front view",
    "sit_wave": "drawn as a full body, sitting on the ground, raising one simple rounded hand to wave",
    "stand_point": "drawn as a full body, standing, pointing forward with one simple rounded hand",
}
PROMPTS = {k: f"{PREAMBLE} Draw {IDENTITY}, now {v}. {CONSTRAINTS}" for k, v in _EDITS.items()}
MODELS = {
    "NB": "google/gemini-2.5-flash-image",
    "NB2": "google/gemini-3.1-flash-image",
    "NBpro": "google/gemini-3-pro-image",
}


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"refs: {len(REFS)} images  |  {len(PROMPTS)} prompts x {len(MODELS)} models")
    tasks = [(pk, mk) for pk in PROMPTS for mk in MODELS]
    for i, (pk, mk) in enumerate(tasks):
        out = os.path.join(OUTDIR, f"{pk}__{mk}.png")
        if os.path.exists(out):
            print(f"[{i + 1}/{len(tasks)}] skip {os.path.basename(out)}")
            continue
        try:
            oe.edit_image(REFS, PROMPTS[pk], MODELS[mk], out)
            print(f"[{i + 1}/{len(tasks)}] ok   {os.path.basename(out)}")
        except Exception as e:  # noqa: BLE001 - one model/call failing shouldn't abort the bake-off
            print(f"[{i + 1}/{len(tasks)}] FAIL {os.path.basename(out)}: {e}")
            traceback.print_exc()

    cols = list(MODELS.keys())
    cell, gut, head = 280, 96, 24
    rows = list(PROMPTS.keys())
    sheet = Image.new("RGB", (gut + len(cols) * cell, head + len(rows) * cell), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((gut + c * cell + cell // 2 - 12, 6), name, fill=(0, 0, 0))
    for r, pk in enumerate(rows):
        y = head + r * cell
        d.text((4, y + cell // 2), pk, fill=(0, 0, 0))
        for c, mk in enumerate(cols):
            p = os.path.join(OUTDIR, f"{pk}__{mk}.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell - 10, cell - 10))
            sheet.paste(im, (gut + c * cell + 5, y + 5))
    grid = os.path.join(OUTDIR, "COMPARISON.jpg")
    sheet.save(grid, quality=90)
    print("wrote", grid)


if __name__ == "__main__":
    main()
