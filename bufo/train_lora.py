"""LoRA fine-tuning of Stable Diffusion on the bufo emoji corpus.

The VAE, text encoder, and UNet base weights stay frozen; only low-rank adapters
on the UNet attention projections train. Each step encodes images to latents,
adds sampled noise at random timesteps, and trains the adapters to predict that
noise (the standard DDPM objective) conditioned on the caption embedding.

Usage:
    uv run python -m bufo.prepare                 # once: build the dataset
    uv run python -m bufo.train_lora --config bufo/configs/lora-sd15.yaml
    # quick smoke: add --max-steps 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mm_training import (
    MetricsLogger,
    RunManifest,
    clip_grad_norm,
    create_optimizer,
    create_scheduler,
    get_device,
    seed_everything,
)
from torch.utils.data import DataLoader

from bufo.config import BufoLoRAConfig, EvalConfig
from bufo.data import BufoDataset
from bufo.eval import generate_eval_images, score_generations
from bufo.pipeline import (
    attach_lora,
    autocast,
    encode_conditioning,
    load_train_components,
    load_training_state,
    save_lora,
    save_training_state,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from bufo.pipeline import TrainComponents


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _noise_target(
    comp: TrainComponents, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """DDPM target: the noise itself (epsilon) or its velocity (v-prediction)."""
    pred_type = comp.noise_scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return comp.noise_scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unsupported prediction_type: {pred_type}")


def diffusion_loss(
    pred: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor, comp: TrainComponents, min_snr_gamma: float
) -> torch.Tensor:
    """MSE loss, optionally min-SNR-weighted (arXiv:2303.09556) when gamma > 0.

    Min-SNR rebalances per-timestep loss so high-noise steps don't dominate —
    faster, more stable convergence.
    """
    if min_snr_gamma <= 0:
        return F.mse_loss(pred.float(), target.float())
    acp = comp.noise_scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snr = acp / (1.0 - acp)
    weight = torch.clamp(snr, max=min_snr_gamma)
    weight = weight / (snr + 1.0) if comp.noise_scheduler.config.prediction_type == "v_prediction" else weight / snr
    per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=list(range(1, pred.ndim)))
    return (weight * per_sample).mean()


def build_live_pipe(comp: TrainComponents) -> object:
    """Inference pipeline over the live (LoRA-adorned) components — for previews + eval."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

    sched = DPMSolverMultistepScheduler.from_config(comp.noise_scheduler.config)
    if comp.base_kind == "sdxl":
        pipe: object = StableDiffusionXLPipeline(
            vae=comp.vae,
            text_encoder=comp.text_encoder,
            text_encoder_2=comp.text_encoder_2,
            tokenizer=comp.tokenizer,
            tokenizer_2=comp.tokenizer_2,
            unet=comp.unet,
            scheduler=sched,
        )
    else:
        pipe = StableDiffusionPipeline(
            vae=comp.vae,
            text_encoder=comp.text_encoder,
            tokenizer=comp.tokenizer,
            unet=comp.unet,
            scheduler=sched,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _guidance(comp: TrainComponents) -> float:
    return 5.0 if comp.base_kind == "sdxl" else 7.5


@torch.no_grad()
def _snapshot(comp: TrainComponents, prompts: list[str], out_dir: Path, device: torch.device, seed: int) -> None:
    """Generate a preview grid from the current LoRA weights (shares live modules)."""
    from PIL import Image

    comp.unet.eval()
    pipe = build_live_pipe(comp)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    images = [pipe(p, num_inference_steps=25, guidance_scale=_guidance(comp), generator=gen).images[0] for p in prompts]
    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for i, im in enumerate(images):
        grid.paste(im, (i * w, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / "grid.png")
    comp.unet.train()


class EvalReporter:
    """Cheap in-training CLIP eval — a few held-out prompts scored every eval_interval.

    Loads CLIP + the train embeddings once; logs ``eval/*`` scalars so quality is
    visible mid-run (alongside ``train/loss``).
    """

    def __init__(self, eval_config: EvalConfig, train_data_dir: str, device: torch.device, logger: object) -> None:
        from bufo.clip_metrics import ClipEmbedder, load_or_build_train_embeddings

        self.cfg = eval_config
        self.logger = logger
        self.embedder = ClipEmbedder.load(eval_config.clip_model, device)
        self.train_emb, self.train_names = load_or_build_train_embeddings(self.embedder, train_data_dir)
        self.subjects = eval_config.step_prompts or eval_config.prompts[:4]
        self.prompts = [eval_config.prompt_template.format(subject=s) + eval_config.suffix for s in self.subjects]

    @torch.no_grad()
    def report(self, comp: TrainComponents, step: int) -> None:
        comp.unet.eval()
        grids = generate_eval_images(
            build_live_pipe(comp),
            self.prompts,
            images_per_prompt=1,
            steps=25,
            guidance=_guidance(comp),
            negative_prompt=self.cfg.negative_prompt,
            seed=self.cfg.seed,
        )
        sc = score_generations(
            grids, self.subjects, self.prompts, self.embedder, self.train_emb, self.train_names, self.cfg, step=step
        )
        comp.unet.train()
        if self.logger is not None:
            keys = ("identity", "prompt_adherence", "legibility", "cartoon", "diversity_overall", "memorization_max")
            for key in keys:
                self.logger.log_scalar(f"eval/{key}", float(getattr(sc, key)), step)
        print(
            f"  eval {step}: identity {sc.identity:.3f} | adher {sc.prompt_adherence:.3f} | "
            f"legib {sc.legibility:.3f} | cartoon {sc.cartoon:+.3f} | memor {sc.memorization_max:.3f}",
            flush=True,
        )


def train(
    config: BufoLoRAConfig,
    *,
    data_dir: str | None = None,
    run_dir: Path | None = None,
    device: torch.device | None = None,
    resume: str | None = None,
    eval_config: EvalConfig | None = None,
) -> Path:
    """Run LoRA fine-tuning. Returns the run directory holding LoRA checkpoints.

    ``resume`` points at a checkpoint dir (or its ``training_state.pt``) to continue
    a run — restores adapters, optimizer, scheduler, step, and RNG. Requires the
    same LoRA config.
    """
    tcfg = config.training
    seed_everything(tcfg.seed)
    device = device or get_device()
    print(f"Device: {device}")

    comp = load_train_components(tcfg.base_model, device, base_kind=tcfg.base_kind)
    trained = attach_lora(comp, config.lora)
    trainable_module = torch.nn.ModuleList(trained)  # optimizer + grad-clip view over all adapters
    total = sum(p.numel() for p in comp.unet.parameters())
    n_train = sum(p.numel() for p in trainable_module.parameters() if p.requires_grad)
    te = "+TE" if config.lora.train_text_encoder else ""
    print(f"Base: {tcfg.base_kind} | UNet {total:,} params | LoRA{te} trainable: {n_train:,}")

    dataset = BufoDataset(
        data_dir or config.data.data_dir,
        comp.tokenizer,
        tokenizer_2=comp.tokenizer_2,
        resolution=config.data.resolution,
        random_flip=config.data.random_flip,
    )
    print(f"Dataset: {len(dataset)} bufos")
    loader = DataLoader(
        dataset,
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=tcfg.num_workers,
    )

    optimizer = create_optimizer(trainable_module, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = create_scheduler(optimizer, warmup_steps=tcfg.warmup_steps, total_steps=tcfg.max_steps)

    resume_step = 0
    if resume:
        rp = Path(resume)
        state_path = rp / "training_state.pt" if rp.is_dir() else rp
        if run_dir is None:
            run_dir = state_path.parent.parent  # checkpoint-N/training_state.pt -> run dir
        resume_step = load_training_state(state_path, comp, optimizer, scheduler, device)
        print(f"Resumed from step {resume_step}: {state_path}")

    logger: MetricsLogger | None = None
    if run_dir is None:
        logger = MetricsLogger(experiment="bufo-lora")
        run_dir = logger.log_dir
    elif resume:
        logger = MetricsLogger(experiment="bufo-lora", run_dir=run_dir)  # keep logging in the same run
    print(f"Run dir: {run_dir}")
    RunManifest.capture(
        experiment="bufo-lora", config=config.model_dump(), seed=tcfg.seed, dataset_id="all-the-bufo"
    ).save(run_dir / "manifest.json")

    reporter = None
    if tcfg.eval_interval and eval_config is not None:
        reporter = EvalReporter(eval_config, data_dir or config.data.data_dir, device, logger)

    comp.unet.train()
    data_iter = _infinite(loader)
    scaling = comp.vae.config.scaling_factor
    n_timesteps = comp.noise_scheduler.config.num_train_timesteps
    t0 = time.time()

    for step in range(resume_step + 1, tcfg.max_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(tcfg.grad_accum):
            batch = next(data_iter)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)

            with torch.no_grad():
                latents = comp.vae.encode(pixel_values).latent_dist.sample() * scaling
            # Outside no_grad so text-encoder LoRA gradients flow (frozen -> no graph).
            cond = encode_conditioning(
                comp, batch, resolution=config.data.resolution, device=device, dtype=latents.dtype
            )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, n_timesteps, (latents.shape[0],), device=device).long()
            noisy = comp.noise_scheduler.add_noise(latents, noise, timesteps)
            target = _noise_target(comp, latents, noise, timesteps)

            with autocast(device, tcfg.amp):
                pred = comp.unet(noisy, timesteps, return_dict=False, **cond)[0]
                loss = diffusion_loss(pred, target, timesteps, comp, tcfg.min_snr_gamma) / tcfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = clip_grad_norm(trainable_module, tcfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            done = step - resume_step
            eta = elapsed / done * (tcfg.max_steps - step) if done else 0.0
            # flush so backgrounded runs show live progress in the captured output
            print(
                f"step {step:>5d}/{tcfg.max_steps} | loss {step_loss:.4f} | "
                f"grad {grad_norm:.2f} | {elapsed:.0f}s | eta {eta:.0f}s",
                flush=True,
            )
        if logger is not None:
            logger.log_scalar("train/loss", step_loss, step)
            logger.log_scalar("train/grad_norm", grad_norm, step)
            logger.log_scalar("train/lr", float(scheduler.get_last_lr()[0]), step)

        if reporter is not None and step % tcfg.eval_interval == 0:
            reporter.report(comp, step)
        if tcfg.snapshot_interval and (step % tcfg.snapshot_interval == 0 or step == tcfg.max_steps):
            print("  rendering snapshot...", flush=True)
            _snapshot(comp, tcfg.snapshot_prompts, run_dir / f"snapshot-{step}", device, tcfg.seed)
        if step % tcfg.checkpoint_interval == 0 or step == tcfg.max_steps:
            ckpt_dir = run_dir / f"checkpoint-{step}"
            save_lora(comp, ckpt_dir)  # inference weights
            save_training_state(comp, optimizer, scheduler, step, ckpt_dir)  # resumable state

    if logger is not None:
        logger.close()
    print(f"Done. Run dir: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="bufo LoRA fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps (quick runs)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint dir or training_state.pt")
    parser.add_argument("--eval-config", type=str, default=None, help="EvalConfig YAML for in-training CLIP eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BufoLoRAConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    eval_config = EvalConfig.from_yaml(args.eval_config) if args.eval_config else None
    train(config, data_dir=args.data_dir, resume=args.resume, eval_config=eval_config)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
