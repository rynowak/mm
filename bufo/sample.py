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

from bufo.data import SUFFIX
from bufo.pipeline import load_inference_pipeline

_DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_DEFAULT_NEG = "photo, realistic, 3d render, cluttered, tiny, text, watermark"
_DEFAULT_PROMPTS = [
    "bufo astronaut",
    "bufo wizard",
    "bufo eating pizza",
    "bufo robot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample bufos from a trained LoRA")
    parser.add_argument("--lora", type=str, default=None, help="LoRA checkpoint dir (omit for base model)")
    parser.add_argument("--base-model", type=str, default=_DEFAULT_BASE)
    parser.add_argument("--base-kind", type=str, default="sd15", choices=["sd15", "sdxl"])
    parser.add_argument("--prompt", action="append", dest="prompts", help="Repeatable; subject phrase")
    parser.add_argument("--out", type=str, default="runs/bufo-samples")
    parser.add_argument("--num", type=int, default=1, help="Images per prompt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", type=str, default=_DEFAULT_NEG)
    parser.add_argument("--raw-prompt", action="store_true", help="Use prompts verbatim (no bufo suffix)")
    parser.add_argument("--emoji", action="append", help="Repeatable shortcode, e.g. :bufo-sip: (uses --rewriter)")
    parser.add_argument("--rewriter", type=str, default="rules", choices=["rules", "llm"])
    parser.add_argument("--rerank", type=int, default=1, help="Generate N candidates/prompt, keep the best by CLIP")
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
    base_kind: str = "sd15",
    negative_prompt: str = _DEFAULT_NEG,
    rerank: int = 1,
    device: torch.device | None = None,
) -> list[Path]:
    """Generate images per prompt and save them. With ``rerank``>1, generate that
    many candidates per prompt and keep the single best by CLIP prompt-adherence."""
    seed_everything(seed)
    device = device or get_device()
    pipe = load_inference_pipeline(base_model, device, Path(lora_dir) if lora_dir else None, base_kind=base_kind)
    pipe.set_progress_bar_config(disable=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = None
    if rerank > 1:
        from bufo.clip_metrics import ClipEmbedder

        embedder = ClipEmbedder.load("openai/clip-vit-base-patch32", device)

    full = [(p + SUFFIX if add_suffix else p) for p in prompts]
    saved: list[Path] = []
    rows: list[list[Image.Image]] = []
    for pi, prompt in enumerate(full):
        n_gen = rerank if rerank > 1 else num
        print(f"  gen {pi + 1}/{len(full)} x{n_gen}: {prompt[:48]}", flush=True)
        cands = [
            pipe(
                prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator(device="cpu").manual_seed(seed + pi * 1000 + n),
            ).images[0]
            for n in range(n_gen)
        ]
        if embedder is not None:  # keep the single best by CLIP prompt-adherence
            sims = embedder.embed_images(cands) @ embedder.embed_texts([prompt])[0]
            cands = [cands[int(sims.argmax())]]
        rows.append(cands)
        for n, im in enumerate(cands):
            path = out_dir / f"bufo_{pi:02d}_{n:02d}.png"
            im.save(path)
            saved.append(path)

    if saved:
        cols = max(len(r) for r in rows)
        w, h = rows[0][0].size
        grid = Image.new("RGB", (w * cols, h * len(rows)), (255, 255, 255))
        for pi, row in enumerate(rows):
            for n, im in enumerate(row):
                grid.paste(im, (n * w, pi * h))
        grid.save(out_dir / "grid.png")
        print(f"  grid -> {out_dir / 'grid.png'}")
    return saved


def main() -> None:
    args = parse_args()
    if args.emoji:  # shortcodes/free-form -> full schema prompts via the rewriter
        from bufo.rewrite import get_rewriter

        rewriter = get_rewriter(args.rewriter)
        prompts = [rewriter.rewrite(e) for e in args.emoji]
        add_suffix = False  # rewriter already produced full schema prompts
    else:
        prompts = args.prompts or _DEFAULT_PROMPTS
        add_suffix = not args.raw_prompt
    generate(
        base_model=args.base_model,
        lora_dir=args.lora,
        prompts=prompts,
        out_dir=Path(args.out),
        num=args.num,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        add_suffix=add_suffix,
        base_kind=args.base_kind,
        negative_prompt=args.negative_prompt,
        rerank=args.rerank,
    )


if __name__ == "__main__":
    main()
