"""Score a bufo checkpoint with CLIP metrics: concept fidelity, prompt adherence,
diversity, memorization, 48px legibility, and cartoon-style margin.

Run with ``--lora <ckpt>`` to score a LoRA, or with no ``--lora`` for the base-model
baseline (the number every later change is measured against).

    uv run python -m bufo.eval --lora runs/bufo-lora/<ts>/checkpoint-500
    uv run python -m bufo.eval                      # base-model baseline
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from mm_training import get_device, seed_everything
from PIL import Image, ImageDraw

from bufo.clip_metrics import (
    CARTOON_TEXT,
    PHOTO_TEXT,
    ClipEmbedder,
    clipscore,
    load_or_build_train_embeddings,
    mean_pairwise_distance,
    nearest_neighbor,
    style_score,
    to_emoji,
    top_k_mean_similarity,
)
from bufo.config import EvalConfig
from bufo.pipeline import load_inference_pipeline

_DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_DEFAULT_EVAL_CONFIG = Path(__file__).resolve().parent / "configs" / "eval-bufo.yaml"


@dataclass
class PromptResult:
    subject: str
    prompt: str
    identity: float  # mean sim to nearest real bufos — "is it our bufo?"
    concept_fidelity: float  # vs generic text — sanity check only
    prompt_adherence: float
    diversity: float
    legibility: float
    cartoon: float
    memorization_max: float
    nearest_train: list[str]
    image_paths: list[str] = field(default_factory=list)


@dataclass
class EvalScorecard:
    checkpoint: str | None
    step: int | None
    clip_model: str
    seed: int
    images_per_prompt: int
    n_prompts: int
    identity: float  # primary "is it a bufo?" — grounded in real bufo images
    concept_fidelity: float  # vs generic text — sanity check only
    prompt_adherence: float
    diversity_overall: float
    diversity_within_prompt: float
    memorization_mean: float
    memorization_max: float
    legibility: float
    cartoon: float
    per_prompt: list[PromptResult]


def generate_eval_images(
    pipe: object,
    prompts: list[str],
    *,
    images_per_prompt: int,
    steps: int,
    guidance: float,
    negative_prompt: str,
    seed: int,
    resolution: int | None = None,
) -> list[list[Image.Image]]:
    """Seeded generation (seed scheme mirrors sample.py). Returns images[prompt][n].

    ``resolution`` should match the LoRA's training resolution: a diffusion model
    generating above the resolution it was trained at duplicates/tiles the subject
    (e.g. an SDXL LoRA trained at 768 tiles when sampled at SDXL's native 1024).
    """
    pipe.set_progress_bar_config(disable=True)
    grids: list[list[Image.Image]] = []
    for pi, prompt in enumerate(prompts):
        print(f"  gen {pi + 1}/{len(prompts)}: {prompt[:48]}", flush=True)
        row: list[Image.Image] = []
        for n in range(images_per_prompt):
            gen = torch.Generator(device="cpu").manual_seed(seed + pi * 1000 + n)
            kwargs = {"num_inference_steps": steps, "guidance_scale": guidance, "generator": gen}
            if resolution:
                kwargs["height"] = kwargs["width"] = resolution
            # FluxPipeline (guidance-distilled) has no `negative_prompt` arg; only pass it to
            # pipelines that accept it (sd15/sdxl), and only when non-empty.
            if negative_prompt and "negative_prompt" in inspect.signature(pipe.__call__).parameters:
                kwargs["negative_prompt"] = negative_prompt
            img = pipe(prompt, **kwargs).images[0]
            row.append(img)
        grids.append(row)
    return grids


def evaluate(
    checkpoint: str | Path | None,
    eval_config: EvalConfig,
    *,
    base_model: str = _DEFAULT_BASE,
    base_kind: str = "sd15",
    device: torch.device | None = None,
    train_data_dir: str = "bufo/data",
    out_dir: Path | None = None,
    subjects: list[str] | None = None,
    images_per_prompt: int | None = None,
    step: int | None = None,
    clip_embedder: ClipEmbedder | None = None,
    write_artifacts: bool = True,
    resolution: int | None = None,
    sampler: str | None = None,
    lora_scale: float = 1.0,
) -> EvalScorecard:
    """Generate the held-out set, score it with CLIP, optionally write artifacts."""
    device = device or get_device()
    seed_everything(eval_config.seed)
    subjects = subjects if subjects is not None else eval_config.prompts
    n_imgs = images_per_prompt if images_per_prompt is not None else eval_config.images_per_prompt
    prompts = [eval_config.prompt_template.format(subject=s) + eval_config.suffix for s in subjects]

    pipe = load_inference_pipeline(
        base_model,
        device,
        Path(checkpoint) if checkpoint else None,
        base_kind=base_kind,
        sampler=sampler,
        lora_scale=lora_scale,
    )
    grids = generate_eval_images(
        pipe,
        prompts,
        images_per_prompt=n_imgs,
        steps=eval_config.num_inference_steps,
        guidance=eval_config.guidance_scale,
        negative_prompt=eval_config.negative_prompt,
        seed=eval_config.seed,
        resolution=resolution,
    )

    embedder = clip_embedder or ClipEmbedder.load(eval_config.clip_model, device)
    train_emb, train_names = load_or_build_train_embeddings(embedder, train_data_dir)
    scorecard = score_generations(
        grids,
        subjects,
        prompts,
        embedder,
        train_emb,
        train_names,
        eval_config,
        step=step,
        checkpoint=str(checkpoint) if checkpoint else None,
    )

    if write_artifacts:
        out_dir = out_dir or _default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_images(grids, scorecard.per_prompt, out_dir)
        write_scorecard(scorecard, out_dir)
        render_contact_sheet(grids, scorecard.per_prompt, scorecard, out_dir / "contact_sheet.png")
        print(f"Wrote scorecard + contact sheet to {out_dir}")
    return scorecard


def score_generations(
    grids: list[list[Image.Image]],
    subjects: list[str],
    prompts: list[str],
    embedder: ClipEmbedder,
    train_emb: torch.Tensor,
    train_names: list[str],
    eval_config: EvalConfig,
    *,
    step: int | None = None,
    checkpoint: str | None = None,
) -> EvalScorecard:
    """Compute the CLIP scorecard for already-generated images (shared by the CLI
    and the in-training reporter)."""
    concept_emb = embedder.embed_texts([eval_config.concept_text])[0]
    cartoon_emb = embedder.embed_texts([CARTOON_TEXT])[0]
    photo_emb = embedder.embed_texts([PHOTO_TEXT])[0]
    prompt_embs = embedder.embed_texts(prompts)

    results: list[PromptResult] = []
    all_full: list[torch.Tensor] = []
    for pi, (subject, prompt, images) in enumerate(zip(subjects, prompts, grids)):  # noqa: B905 — py3.9 has no strict=
        full = embedder.embed_images(images)
        small = embedder.embed_images([to_emoji(im) for im in images])
        all_full.append(full)
        nn_vals, nn_idx = nearest_neighbor(full, train_emb)
        results.append(
            PromptResult(
                subject=subject,
                prompt=prompt,
                identity=top_k_mean_similarity(full, train_emb).mean().item(),
                concept_fidelity=clipscore(full @ concept_emb, eval_config.clipscore_w).mean().item(),
                prompt_adherence=clipscore(full @ prompt_embs[pi], eval_config.clipscore_w).mean().item(),
                diversity=mean_pairwise_distance(full),
                legibility=clipscore(small @ concept_emb, eval_config.clipscore_w).mean().item(),
                cartoon=style_score(full, cartoon_emb, photo_emb).mean().item(),
                memorization_max=nn_vals.max().item(),
                nearest_train=[train_names[i] for i in nn_idx.tolist()],
            )
        )

    all_emb = torch.cat(all_full)
    nn_all, _ = nearest_neighbor(all_emb, train_emb)
    return EvalScorecard(
        checkpoint=checkpoint,
        step=step,
        clip_model=eval_config.clip_model,
        seed=eval_config.seed,
        images_per_prompt=len(grids[0]) if grids else 0,
        n_prompts=len(subjects),
        identity=top_k_mean_similarity(all_emb, train_emb).mean().item(),
        concept_fidelity=_mean(r.concept_fidelity for r in results),
        prompt_adherence=_mean(r.prompt_adherence for r in results),
        diversity_overall=mean_pairwise_distance(all_emb),
        diversity_within_prompt=_mean(r.diversity for r in results),
        memorization_mean=nn_all.mean().item(),
        memorization_max=nn_all.max().item(),
        legibility=_mean(r.legibility for r in results),
        cartoon=_mean(r.cartoon for r in results),
        per_prompt=results,
    )


def _mean(xs) -> float:
    vals = list(xs)
    return sum(vals) / len(vals) if vals else 0.0


def _default_out_dir() -> Path:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("runs/bufo-eval") / ts


def _save_images(grids: list[list[Image.Image]], results: list[PromptResult], out_dir: Path) -> None:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for pi, row in enumerate(grids):
        for n, im in enumerate(row):
            path = img_dir / f"{pi:02d}_{n:02d}.png"
            im.save(path)
            results[pi].image_paths.append(str(path))


def write_scorecard(scorecard: EvalScorecard, out_dir: Path) -> None:
    (out_dir / "scorecard.json").write_text(json.dumps(dataclasses.asdict(scorecard), indent=2))


def render_contact_sheet(
    grids: list[list[Image.Image]], results: list[PromptResult], scorecard: EvalScorecard, out_path: Path
) -> None:
    """One row per prompt (label gutter + its images), header with overall scores."""
    if not grids or not grids[0]:
        return
    # Thumbnail each cell — full-res (e.g. SDXL 1024) makes a 24k-px sheet that
    # viewers render as blank.
    cell = 224
    gutter, header = 240, 36
    cols = max(len(r) for r in grids)
    width = gutter + cols * cell
    height = header + len(grids) * cell
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 12),
        f"identity {scorecard.identity:.3f} | adher {scorecard.prompt_adherence:.3f} | "
        f"divers {scorecard.diversity_overall:.3f} | legib@48 {scorecard.legibility:.3f} | "
        f"cartoon {scorecard.cartoon:+.3f} | memor_max {scorecard.memorization_max:.3f}",
        fill=(0, 0, 0),
    )
    for pi, (row, res) in enumerate(zip(grids, results)):  # noqa: B905 — py3.9 has no strict=
        y = header + pi * cell
        label = (
            f"{res.subject}\nidentity {res.identity:.2f}  adher {res.prompt_adherence:.2f}\n"
            f"legib {res.legibility:.2f}  cartoon {res.cartoon:+.2f}\nmemor {res.memorization_max:.2f}"
        )
        draw.multiline_text((8, y + 8), label, fill=(0, 0, 0), spacing=3)
        for n, im in enumerate(row):
            sheet.paste(im.resize((cell, cell)), (gutter + n * cell, y))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP-score a bufo checkpoint")
    parser.add_argument("--lora", type=str, default=None, help="LoRA checkpoint dir (omit for base baseline)")
    parser.add_argument("--base-model", type=str, default=_DEFAULT_BASE)
    parser.add_argument("--base-kind", type=str, default="sd15", choices=["sd15", "sdxl", "flux"])
    parser.add_argument("--eval-config", type=str, default=str(_DEFAULT_EVAL_CONFIG))
    parser.add_argument("--data-dir", type=str, default="bufo/data")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--images-per-prompt", type=int, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Sampling resolution; set to the LoRA's training resolution to avoid subject tiling (e.g. 768 for SDXL).",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        help="Override sampler (sd15/sdxl): dpmpp_2m_karras | dpmpp_sde_karras | euler. Default = base default.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Strength to fuse the LoRA at (<1.0 tames high-rank over-application / subject tiling).",
    )
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.eval_config)
    scorecard = evaluate(
        args.lora,
        cfg,
        base_model=args.base_model,
        base_kind=args.base_kind,
        train_data_dir=args.data_dir,
        out_dir=Path(args.out) if args.out else None,
        images_per_prompt=args.images_per_prompt,
        resolution=args.resolution,
        sampler=args.sampler,
        lora_scale=args.lora_scale,
    )
    print(
        f"identity {scorecard.identity:.3f} | adherence {scorecard.prompt_adherence:.3f} | "
        f"diversity {scorecard.diversity_overall:.3f} | legibility@48 {scorecard.legibility:.3f} | "
        f"cartoon {scorecard.cartoon:+.3f} | memorization_max {scorecard.memorization_max:.3f} | "
        f"concept(text) {scorecard.concept_fidelity:.3f}"
    )


if __name__ == "__main__":
    main()
