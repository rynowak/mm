"""Generate the bufo eval grid from an SD3.5 base + a trained LoRA (for the Large swing).

Loads the base pipeline (default SD3.5-Large), applies the LoRA weights from LORA_DIR, and
renders the same 16-prompt adult-anchored grid as sd35_gen_grid.py so the 8B-LoRA result can
be compared by eye against the Medium full-FT. No judge metric (DINOv2 retired).

Env: BASE (def SD3.5-large), LORA_DIR, OUT, RES (def 1024), IPP (def 2), LORA_SCALE (def 1.0).
"""

from __future__ import annotations

import os

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

BASE = os.environ.get("BASE", "stabilityai/stable-diffusion-3.5-large")
LORA_DIR = os.environ["LORA_DIR"]
OUT = os.environ["OUT"]
RES = int(os.environ.get("RES", "1024"))
IPP = int(os.environ.get("IPP", "2"))
LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))

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

    pipe = StableDiffusion3Pipeline.from_pretrained(BASE, torch_dtype=torch.bfloat16).to(dev)
    pipe.load_lora_weights(LORA_DIR)
    try:
        pipe.set_adapters(pipe.get_active_adapters(), adapter_weights=[LORA_SCALE])
    except Exception as e:  # noqa: BLE001 - scale is best-effort; default 1.0 already applied
        print("note: could not set lora scale:", e)
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
