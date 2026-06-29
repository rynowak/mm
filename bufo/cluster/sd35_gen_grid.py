"""Generate the standard bufo eval grid from one SD3.5 transformer checkpoint (no judge).

Lean generate-only version used by the step-sweep: loads base + the given transformer,
renders the 16-prompt adult-anchored grid at IPP per prompt, and writes a contact sheet +
individual images. DINOv2 is intentionally dropped (proven not to track the user's eye).

Env: BASE, FT (transformer dir), OUT, RES (def 1024), IPP (def 2).
"""

from __future__ import annotations

import os

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image

BASE = os.environ.get("BASE", "stabilityai/stable-diffusion-3.5-medium")
FT = os.environ["FT"]
OUT = os.environ["OUT"]
RES = int(os.environ.get("RES", "1024"))
IPP = int(os.environ.get("IPP", "2"))

SUBJECTS = [
    "neutral",
    "happy",
    "sad",
    "crying",
    "angry",
    "smug",
    "surprised",
    "content",
    "happy, holding coffee",
    "neutral, wearing a wizard hat",
    "angry, holding a torch",
    "content, holding flowers",
    "happy, eating pizza",
    "neutral, holding a book",
    "concerned, sitting",
    "happy, arms raised",
]
PROMPT = "olive green adult bufo, {s}, soft-shaded cartoon sticker"
NEG = "deformed, blurry, low quality, extra limbs, teeth, fangs, text, watermark"


def main() -> None:
    img_out = os.path.join(OUT, "images")
    os.makedirs(img_out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise SystemExit("no CUDA")

    transformer = SD3Transformer2DModel.from_pretrained(FT, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    paths: list[str] = []
    for i, s in enumerate(SUBJECTS):
        for j in range(IPP):
            g = torch.Generator(device=dev).manual_seed(1000 + j)
            img = pipe(
                prompt=PROMPT.format(s=s),
                negative_prompt=NEG,
                num_inference_steps=28,
                guidance_scale=4.5,
                height=RES,
                width=RES,
                generator=g,
            ).images[0]
            p = os.path.join(img_out, f"{i:02d}_{j}.png")
            img.save(p)
            paths.append(p)
        print(f"gen {i + 1}/{len(SUBJECTS)}: {s}", flush=True)

    cols, cell, rows = IPP, 360, len(SUBJECTS)
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for idx, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 8, cell - 8))
        sheet.paste(im, ((idx % cols) * cell + 4, (idx // cols) * cell + 4))
    sheet.save(os.path.join(OUT, "contact_sheet.png"))
    print("DONE", os.path.join(OUT, "contact_sheet.png"), flush=True)


if __name__ == "__main__":
    main()
