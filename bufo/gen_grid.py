"""Generate a quick contact grid from a trained LoRA — no CLIP scoring, just eyeballing.

Lighter than bufo.eval (no embedder / train embeddings), so it fits the GPU pod's
host-RAM cap more easily. Used to sanity-check Flux output quality + identity.

    python -m bufo.gen_grid --lora <ckpt> --base-kind flux --out /mnt/ray/bufo-eval/flux_grid.png
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from mm_training import get_device
from PIL import Image

from bufo.pipeline import FLUX_DEFAULT_BASE, load_inference_pipeline

# Includes the two prompts SDXL mangled (bicycle, bubble tea) + identity checks.
PROMPTS = [
    "bufo riding a skateboard",
    "bufo riding a bicycle",
    "bufo drinking bubble tea",
    "bufo as a ninja",
    "bufo offering a flower",
    "bufo crying",
]
SUFFIX = ", flat cartoon frog emoji sticker, bold simple shapes, white background"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a quick LoRA contact grid (no scoring)")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base-kind", default="flux")
    ap.add_argument("--base-model", default=FLUX_DEFAULT_BASE)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    device = get_device()
    pipe = load_inference_pipeline(args.base_model, device, pathlib.Path(args.lora), base_kind=args.base_kind)
    pipe.set_progress_bar_config(disable=True)

    images = []
    for i, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(1000 + i)
        img = pipe(prompt + SUFFIX, num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen).images[
            0
        ]
        images.append(img)
        print(f"gen {i + 1}/{len(PROMPTS)}: {prompt}", flush=True)

    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for i, im in enumerate(images):
        sheet.paste(im, (i * w, 0))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
