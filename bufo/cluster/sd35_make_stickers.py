"""Generate a transparent-background bufo emoji STICKER SET from the SD3.5 full-FT model.

The deliverable: for each emoji in a curated vocabulary (emotions, hand gestures, props,
costumes), generate CAND candidates with the winning prompt schema, cut out the background
(rembg), auto-trim to content + center on a square canvas, and export a 1024 master + a
128px Slack-ready PNG (both RGBA).

Incremental by design (AGENTS.md): every sticker is written as soon as it's made, the run
resumes by skipping masters that already exist, and progress streams with an ETA. A crash
loses at most the in-flight image.

Env: FT (transformer dir), BASE, OUT, RES (gen res, def 1024), CAND (def 4), SLACK (def 128),
EMOJI_LIMIT (process only first N emojis; for smoke tests).
"""

from __future__ import annotations

import json
import os
import time

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image, ImageDraw

BASE = os.environ.get("BASE", "stabilityai/stable-diffusion-3.5-medium")
FT = os.environ.get("FT", "/mnt/ray/bufo-keep/sd35-medium-ft-1000/transformer")
OUT = os.environ.get("OUT", "/mnt/ray/bufo-runs/sd35-medium-ft/stickers")
RES = int(os.environ.get("RES", "1024"))
CAND = int(os.environ.get("CAND", "4"))
SLACK = int(os.environ.get("SLACK", "128"))
EMOJI_LIMIT = int(os.environ.get("EMOJI_LIMIT", "0"))

PROMPT = "olive green adult bufo, {d}, soft-shaded cartoon sticker"
NEG = "deformed, blurry, low quality, extra limbs, teeth, fangs, text, watermark, photo, realistic"

# (name, descriptor) — bufo emoji vocabulary across all four categories. name is the slug.
EMOJIS: list[tuple[str, str]] = [
    # --- emotions / reactions ---
    ("happy", "happy, smiling"),
    ("grin", "big toothless grin, very happy"),
    ("sad", "sad, frowning"),
    ("cry", "crying, a single tear"),
    ("sob", "sobbing, streams of tears"),
    ("angry", "angry, furrowed brow"),
    ("rage", "furious, red-faced, steam from head"),
    ("love", "heart-shaped eyes, in love"),
    ("laugh", "laughing hard, eyes closed"),
    ("smug", "smug, smirking"),
    ("surprised", "surprised, wide eyes"),
    ("shocked", "shocked, mouth open"),
    ("confused", "confused, head tilted"),
    ("think", "thinking, hand on chin"),
    ("sleepy", "sleepy, half-closed eyes"),
    ("cool", "wearing sunglasses, looking cool"),
    ("nervous", "nervous, single sweat drop"),
    ("pleading", "pleading, big teary puppy eyes"),
    ("dead", "x for eyes, fainted"),
    ("wink", "winking, one eye closed, tongue out"),
    ("blush", "blushing, shy smile"),
    ("scared", "scared, wide frightened eyes"),
    ("bored", "bored, unamused flat expression"),
    ("mindblown", "mind blown, amazed expression"),
    # --- hand gestures ---
    ("thumbsup", "giving a thumbs up"),
    ("thumbsdown", "giving a thumbs down"),
    ("wave", "waving hello, one hand raised"),
    ("salute", "saluting"),
    ("facepalm", "facepalm, hand covering face"),
    ("pray", "praying, hands together"),
    ("clap", "clapping hands"),
    ("ok", "making an ok hand sign"),
    ("shrug", "shrugging, both palms up"),
    # --- props / objects ---
    ("coffee", "holding a coffee mug"),
    ("heart", "holding a big red heart"),
    ("fire", "holding a lit torch"),
    ("pizza", "holding a slice of pizza"),
    ("cake", "holding a birthday cake"),
    ("flowers", "holding a bouquet of flowers"),
    ("book", "reading an open book"),
    ("phone", "looking at a smartphone"),
    ("popcorn", "holding a bucket of popcorn"),
    ("money", "holding a stack of cash"),
    ("gaming", "holding a game controller"),
    ("balloon", "holding a balloon"),
    ("gift", "holding a wrapped gift box"),
    ("music", "wearing headphones, music notes"),
    # --- costumes / hats ---
    ("wizard", "wearing a wizard hat"),
    ("crown", "wearing a golden crown"),
    ("party", "wearing a party hat, confetti"),
    ("chef", "wearing a chef hat"),
    ("graduate", "wearing a graduation cap"),
    ("detective", "wearing a detective hat, holding magnifying glass"),
    ("santa", "wearing a santa hat"),
    ("cowboy", "wearing a cowboy hat"),
]


def cutout(img: Image.Image, size: int) -> Image.Image:
    """rembg cutout -> trim to alpha bbox -> center on a square transparent canvas."""
    from rembg import remove

    rgba = remove(img.convert("RGBA"))
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    side = max(rgba.size)
    margin = int(side * 0.08)
    canvas = Image.new("RGBA", (side + 2 * margin, side + 2 * margin), (0, 0, 0, 0))
    canvas.paste(rgba, ((canvas.width - rgba.width) // 2, (canvas.height - rgba.height) // 2), rgba)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    masters = os.path.join(OUT, "masters")
    slack = os.path.join(OUT, "slack")
    os.makedirs(masters, exist_ok=True)
    os.makedirs(slack, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} FT={FT} OUT={OUT} CAND={CAND} limit={EMOJI_LIMIT}", flush=True)
    if dev.type != "cuda":
        raise SystemExit("no CUDA — refusing to generate on CPU (too slow)")

    emojis = EMOJIS[:EMOJI_LIMIT] if EMOJI_LIMIT > 0 else EMOJIS

    transformer = SD3Transformer2DModel.from_pretrained(FT, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    jobs = [(name, d, c) for (name, d) in emojis for c in range(CAND)]
    todo = [(n, d, c) for (n, d, c) in jobs if not os.path.exists(os.path.join(masters, f"{n}_{c}.png"))]
    print(f"total={len(jobs)} todo={len(todo)} (resuming, {len(jobs) - len(todo)} done)", flush=True)

    man = open(os.path.join(OUT, "manifest.jsonl"), "a")  # noqa: SIM115 (streaming append across loop)
    t0 = time.time()
    for i, (name, d, c) in enumerate(todo):
        g = torch.Generator(device=dev).manual_seed(100 + c)
        img = pipe(
            prompt=PROMPT.format(d=d),
            negative_prompt=NEG,
            num_inference_steps=28,
            guidance_scale=4.5,
            height=RES,
            width=RES,
            generator=g,
        ).images[0]
        master = cutout(img, RES)
        mp = os.path.join(masters, f"{name}_{c}.png")
        sp = os.path.join(slack, f"{name}_{c}.png")
        master.save(mp)
        cutout(img, SLACK).save(sp)
        man.write(json.dumps({"name": name, "cand": c, "desc": d, "master": mp, "slack": sp}) + "\n")
        man.flush()
        done = i + 1
        rate = (time.time() - t0) / done
        eta = rate * (len(todo) - done)
        print(f"[{done}/{len(todo)}] {name}_{c}  {rate:.1f}s/img  eta {eta / 60:.1f}m", flush=True)
    man.close()

    # contact sheet: one row per emoji, CAND candidates across, composited on grey + label.
    cell = 240
    label_h = 22
    rows = len(emojis)
    sheet = Image.new("RGB", (CAND * cell, rows * (cell + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (name, _) in enumerate(emojis):
        for c in range(CAND):
            mp = os.path.join(masters, f"{name}_{c}.png")
            if not os.path.exists(mp):
                continue
            im = Image.open(mp).convert("RGBA")
            im.thumbnail((cell - 12, cell - 12))
            tile = Image.new("RGB", (cell, cell), (235, 235, 235))
            tile.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
            sheet.paste(tile, (c * cell, r * (cell + label_h)))
        draw.text((4, r * (cell + label_h) + cell + 4), name, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "contact_sheet.png"))
    print("DONE wrote", os.path.join(OUT, "contact_sheet.png"), flush=True)


if __name__ == "__main__":
    main()
