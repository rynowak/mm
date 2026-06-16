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

from bufo.config import BufoLoRAConfig
from bufo.data import BufoDataset
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


@torch.no_grad()
def _snapshot(comp: TrainComponents, prompts: list[str], out_dir: Path, device: torch.device, seed: int) -> None:
    """Generate a preview grid from the current LoRA weights (shares live modules)."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
    from PIL import Image

    comp.unet.eval()
    sched = DPMSolverMultistepScheduler.from_config(comp.noise_scheduler.config)
    if comp.base_kind == "sdxl":
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline = StableDiffusionXLPipeline(
            vae=comp.vae,
            text_encoder=comp.text_encoder,
            text_encoder_2=comp.text_encoder_2,
            tokenizer=comp.tokenizer,
            tokenizer_2=comp.tokenizer_2,
            unet=comp.unet,
            scheduler=sched,
        )
        guidance = 5.0
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
        guidance = 7.5
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    images = [pipe(p, num_inference_steps=25, guidance_scale=guidance, generator=gen).images[0] for p in prompts]
    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h), (255, 255, 255))
    for i, im in enumerate(images):
        grid.paste(im, (i * w, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / "grid.png")
    comp.unet.train()


def train(
    config: BufoLoRAConfig,
    *,
    data_dir: str | None = None,
    run_dir: Path | None = None,
    device: torch.device | None = None,
    resume: str | None = None,
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
                loss = F.mse_loss(pred.float(), target.float()) / tcfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = clip_grad_norm(trainable_module, tcfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:>5d}/{tcfg.max_steps} | loss {step_loss:.4f} | grad {grad_norm:.2f} | {elapsed:.0f}s")
        if logger is not None:
            logger.log_scalar("train/loss", step_loss, step)
            logger.log_scalar("train/grad_norm", grad_norm, step)
            logger.log_scalar("train/lr", float(scheduler.get_last_lr()[0]), step)

        if tcfg.snapshot_interval and (step % tcfg.snapshot_interval == 0 or step == tcfg.max_steps):
            print("  rendering snapshot...")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BufoLoRAConfig.from_yaml(args.config)
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    train(config, data_dir=args.data_dir, resume=args.resume)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, lambda *_: (print("\nInterrupted."), exit(0)))
    main()
