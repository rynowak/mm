"""Generate new bufos from a trained LoRA checkpoint.

Usage:
    uv run python -m bufo.sample --lora runs/bufo-lora/<ts>/checkpoint-1500 \\
        --prompt "a bufo of astronaut" --prompt "a bufo of pizza" --num 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mm_training import get_device, seed_everything
from PIL import Image

from bufo.pipeline import load_inference_pipeline

_DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_SUFFIX = ", frog emoji sticker, white background"
_DEFAULT_PROMPTS = [
    "a bufo of astronaut",
    "a bufo of wizard",
    "a bufo of pizza",
    "a bufo of robot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample bufos from a trained LoRA")
    parser.add_argument("--lora", type=str, default=None, help="LoRA checkpoint dir (omit for base model)")
    parser.add_argument("--base-model", type=str, default=_DEFAULT_BASE)
    parser.add_argument("--prompt", action="append", dest="prompts", help="Repeatable; subject phrase")
    parser.add_argument("--out", type=str, default="runs/bufo-samples")
    parser.add_argument("--num", type=int, default=1, help="Images per prompt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-prompt", action="store_true", help="Use prompts verbatim (no bufo suffix)")
    return parser.parse_args()


def generate(
    *,
    base_model: str,
    lora_dir: str | None,
    prompts: list[str],
    out_dir: Path,
    num: int,
    steps: int,
    guidance: float,
    seed: int,
    add_suffix: bool,
    device: torch.device | None = None,
) -> list[Path]:
    """Generate ``num`` images per prompt and save them. Returns saved paths."""
    seed_everything(seed)
    device = device or get_device()
    pipe = load_inference_pipeline(base_model, device, Path(lora_dir) if lora_dir else None)
    pipe.set_progress_bar_config(disable=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = [(p + _SUFFIX if add_suffix else p) for p in prompts]
    saved: list[Path] = []
    images: list[Image.Image] = []
    for pi, prompt in enumerate(full):
        for n in range(num):
            gen = torch.Generator(device="cpu").manual_seed(seed + pi * 1000 + n)
            img = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]
            path = out_dir / f"bufo_{pi:02d}_{n:02d}.png"
            img.save(path)
            saved.append(path)
            images.append(img)
            print(f"  {path}  <- {prompt!r}")

    if images:
        w, h = images[0].size
        cols = num
        rows = len(prompts)
        grid = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
        for idx, im in enumerate(images):
            grid.paste(im, ((idx % cols) * w, (idx // cols) * h))
        grid_path = out_dir / "grid.png"
        grid.save(grid_path)
        print(f"  grid -> {grid_path}")
    return saved


def main() -> None:
    args = parse_args()
    prompts = args.prompts or _DEFAULT_PROMPTS
    generate(
        base_model=args.base_model,
        lora_dir=args.lora,
        prompts=prompts,
        out_dir=Path(args.out),
        num=args.num,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        add_suffix=not args.raw_prompt,
    )


if __name__ == "__main__":
    main()
