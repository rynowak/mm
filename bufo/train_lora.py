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


# ----------------------------------------------------------------------------
# FLUX.1 flow-matching (rectified flow)
#
# All math below mirrors diffusers' examples/dreambooth/train_dreambooth_lora_flux.py
# (diffusers >= 0.30; verified against v0.38.0) and FluxPipeline static helpers, so
# it stays auditable against the reference. Function names from the reference are
# cited inline.
# ----------------------------------------------------------------------------


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack 2x2 latent patches into a token sequence. Mirrors FluxPipeline._pack_latents.

    [B, C, H, W] -> [B, (H/2)*(W/2), C*4]. H and W must be even (true for SD/flux
    VAE latents at any /16 image resolution).
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def _prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Positional ids for the packed image tokens. Mirrors FluxPipeline._prepare_latent_image_ids.

    ``height``/``width`` are the *packed* grid dims (latent H/2, W/2). Returns
    [height*width, 3]: channel 0 is zero, channels 1/2 are the row/col index.
    """
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    h, w, c = latent_image_ids.shape
    return latent_image_ids.reshape(h * w, c).to(device=device, dtype=dtype)


def _sd3_loss_weighting(scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """SD3/flux loss weighting hook. Mirrors diffusers compute_loss_weighting_for_sd3.

    ``sigmas`` is the per-sample sigma broadcastable to the latent. "none" (default)
    is uniform weighting — start here; the other schemes are available as a hook.
    """
    import math

    if scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * bot)
    return torch.ones_like(sigmas)


def sample_flux_sigmas(batch_size: int, logit_mean: float, logit_std: float, device: torch.device) -> torch.Tensor:
    """Per-sample timestep density t in (0, 1) via logit-normal sampling.

    Mirrors diffusers compute_density_for_timestep_sampling(weighting_scheme=
    "logit_normal"): t = sigmoid(normal(logit_mean, logit_std)). For the default
    FlowMatchEulerDiscreteScheduler, sigma == t exactly (scheduler sigma =
    timestep / num_train_timesteps), so we use t directly as the flow sigma and
    pass timesteps = t * 1000 to the transformer. This is the continuous form of
    the reference's `indices = (u * num_train_timesteps).long()` quantization.
    """
    u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
    return torch.sigmoid(u)


def flux_flow_match_loss(comp: TrainComponents, batch: dict, cfg: object, device: torch.device) -> torch.Tensor:
    """One flux flow-matching training step. Returns the scalar (unscaled) loss.

    Rectified-flow objective, mirroring the train_dreambooth_lora_flux.py loop:

      latents = (vae.encode(x).sample - shift_factor) * scaling_factor
      t       = sigmoid(normal(logit_mean, logit_std))         # logit-normal density
      sigma   = t                                              # flow sigma
      noisy   = (1 - sigma) * latents + sigma * noise          # zt = (1-t)x + t*z1
      target  = noise - latents                                # velocity (z1 - x)
      pred    = transformer(pack(noisy), timestep=t, guidance=g, ...) -> unpack
      loss    = mean( w(sigma) * (pred - target)^2 )           # w = SD3 weighting (none by default)

    The transformer is fed the 2x2-packed latents + latent_image_ids; text comes in
    as pooled CLIP (pooled_projections) + T5 sequence (encoder_hidden_states) + txt_ids.
    """
    from diffusers import FluxPipeline

    vae = comp.vae
    transformer = comp.unet
    shift_factor = vae.config.shift_factor
    scaling_factor = vae.config.scaling_factor

    # Encode at the VAE's own dtype (bf16 for flux) — feeding fp32 pixels into a bf16
    # VAE is a dtype-mismatch crash. Flux's VAE is bf16-safe (unlike SDXL's fp16 VAE).
    pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
    # diffusers: model_input = (model_input - shift_factor) * scaling_factor
    latents = (latents - shift_factor) * scaling_factor

    # Conditioning (outside no_grad in principle; encoders frozen -> no graph).
    cond = encode_conditioning(comp, batch, resolution=0, device=device, dtype=latents.dtype)

    bsz, channels, lat_h, lat_w = latents.shape
    noise = torch.randn_like(latents)

    # Logit-normal timestep density; sigma == t for the default flow scheduler.
    t = sample_flux_sigmas(bsz, cfg.flux_logit_mean, cfg.flux_logit_std, device).to(dtype=latents.dtype)
    sigma = t.view(bsz, 1, 1, 1)  # broadcast over [B, C, H, W]
    noisy = (1.0 - sigma) * latents + sigma * noise  # zt = (1 - t) x + t z1

    packed_noisy = _pack_latents(noisy)
    latent_image_ids = _prepare_latent_image_ids(lat_h // 2, lat_w // 2, device, latents.dtype)

    # Guidance: FLUX.1-dev is guidance-distilled (config.guidance_embeds True) and
    # needs an embedded guidance vector; schnell (guidance_embeds False) takes None.
    if getattr(transformer.config, "guidance_embeds", False):
        guidance = torch.full((bsz,), float(cfg.flux_guidance), device=device, dtype=latents.dtype)
    else:
        guidance = None

    with autocast(device, cfg.amp):
        model_pred = transformer(
            hidden_states=packed_noisy,
            timestep=t,  # reference passes timesteps/1000 == t (transformer rescales by 1000 internally)
            guidance=guidance,
            pooled_projections=cond["pooled_projections"],
            encoder_hidden_states=cond["encoder_hidden_states"],
            txt_ids=cond["txt_ids"],
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        # _unpack_latents back to [B, C, H, W] so it aligns with target.
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        model_pred = FluxPipeline._unpack_latents(
            model_pred,
            height=lat_h * vae_scale_factor,
            width=lat_w * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        # flow-matching target = velocity = noise - latents (== z1 - x).
        target = noise - latents
        weighting = _sd3_loss_weighting(cfg.flux_weighting_scheme, sigma)
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1), dim=1
        ).mean()
    return loss


def build_live_pipe(comp: TrainComponents) -> object:
    """Inference pipeline over the live (LoRA-adorned) components — for previews + eval."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

    if comp.base_kind == "flux":
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline

        flux_pipe = FluxPipeline(
            vae=comp.vae,
            text_encoder=comp.text_encoder,
            text_encoder_2=comp.text_encoder_2,
            tokenizer=comp.tokenizer,
            tokenizer_2=comp.tokenizer_2,
            transformer=comp.unet,
            scheduler=FlowMatchEulerDiscreteScheduler(),
        )
        flux_pipe.set_progress_bar_config(disable=True)
        return flux_pipe

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
    if comp.base_kind == "flux":
        return 3.5  # FLUX.1-dev inference default (embedded distilled guidance)
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

    # Flux (12B transformer + 4.7B T5) must load in bf16 to fit an A100 80GB; in fp32
    # the frozen base weights alone are ~68GB and OOM. LoRA adapter params stay fp32
    # (forced in attach_lora). sd15/sdxl keep fp32 weights + amp autocast as before.
    load_dtype = torch.bfloat16 if tcfg.base_kind == "flux" else torch.float32
    comp = load_train_components(tcfg.base_model, device, base_kind=tcfg.base_kind, dtype=load_dtype)
    trained = attach_lora(comp, config.lora)
    # Gradient checkpointing on the flux transformer — required to fit 1024px on an
    # A100 80GB (trades recompute for activation memory).
    if tcfg.base_kind == "flux" and tcfg.flux_gradient_checkpointing:
        comp.unet.enable_gradient_checkpointing()
    trainable_module = torch.nn.ModuleList(trained)  # optimizer + grad-clip view over all adapters
    total = sum(p.numel() for p in comp.unet.parameters())
    n_train = sum(p.numel() for p in trainable_module.parameters() if p.requires_grad)
    te = "+TE" if config.lora.train_text_encoder else ""
    denoiser = "transformer" if tcfg.base_kind == "flux" else "UNet"
    print(f"Base: {tcfg.base_kind} | {denoiser} {total:,} params | LoRA{te} trainable: {n_train:,}")

    dataset = BufoDataset(
        data_dir or config.data.data_dir,
        comp.tokenizer,
        tokenizer_2=comp.tokenizer_2,
        resolution=config.data.resolution,
        random_flip=config.data.random_flip,
        base_kind=tcfg.base_kind,
        tokenizer_2_max_length=tcfg.flux_max_sequence_length,
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
    is_flux = tcfg.base_kind == "flux"
    scaling = comp.vae.config.scaling_factor
    t0 = time.time()

    for step in range(resume_step + 1, tcfg.max_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(tcfg.grad_accum):
            batch = next(data_iter)

            if is_flux:
                # Rectified-flow path: VAE encode + pack + flow-match loss, all inside
                # flux_flow_match_loss (mirrors train_dreambooth_lora_flux.py).
                loss = flux_flow_match_loss(comp, batch, tcfg, device) / tcfg.grad_accum
                loss.backward()
                step_loss += loss.item()
                continue

            assert comp.noise_scheduler is not None  # DDPM path (sd15/sdxl) always has one
            n_timesteps = comp.noise_scheduler.config.num_train_timesteps
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
