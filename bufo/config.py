"""Pydantic config models for the bufo LoRA fine-tuning sample."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path


class DataConfig(BaseModel):
    # Public source: github.com/knobiknows/all-the-bufo (PNGs under all-the-bufo/).
    repo: str = "knobiknows/all-the-bufo"
    ref: str = "main"
    subdir: str = "all-the-bufo"
    data_dir: str = "bufo/data"  # holds raw/, images/, metadata.jsonl
    resolution: int = 512
    # Drop the bigbufo_<r>_<c> tiles — they are slices of one giant bufo, not
    # standalone emoji, and would teach the model partial crops.
    exclude_substrings: list[str] = ["bigbufo_"]
    random_flip: bool = True
    crop: bool = False  # center-crop to square (tighter) vs pad (default, lossless)


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 16  # scaling = alpha / rank
    dropout: float = 0.0
    # SD UNet cross/self-attention projection names (peft matches by suffix).
    # For flux, set these to the FluxTransformer2DModel attention projections
    # (see flux_target_modules below for the canonical set).
    target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    # Also adapt the CLIP text encoder(s) — the biggest prompt-adherence lever.
    # (sd15/sdxl only; flux leaves both text encoders frozen.)
    train_text_encoder: bool = False
    text_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "out_proj"]
    # FluxTransformer2DModel attention projections (single + double stream blocks).
    # Used by attach_lora when base_kind == "flux"; overridable per-config.
    flux_target_modules: list[str] = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ]


class TrainingConfig(BaseModel):
    seed: int = 42
    base_kind: Literal["sd15", "sdxl", "flux"] = "sd15"
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    batch_size: int = 1
    grad_accum: int = 4  # effective batch = batch_size * grad_accum
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    max_steps: int = 1500
    grad_clip: float = 1.0
    min_snr_gamma: float = 0.0  # >0 enables min-SNR loss weighting (5.0 is typical, sd15/sdxl only)
    # Flux flow-matching knobs (ignored for sd15/sdxl). The logit-normal timestep
    # sampler + guidance mirror diffusers' train_dreambooth_lora_flux.py.
    flux_logit_mean: float = 0.0  # compute_density_for_timestep_sampling(logit_mean=)
    flux_logit_std: float = 1.0  # compute_density_for_timestep_sampling(logit_std=)
    flux_weighting_scheme: str = "none"  # SD3 loss weighting hook: none|sigma_sqrt|cosmap|logit_normal
    flux_guidance: float = 1.0  # embedded guidance for FLUX.1-dev (guidance-distilled); ignored for schnell
    flux_max_sequence_length: int = 512  # T5 token budget (256 fits more easily; 512 = dev default)
    flux_gradient_checkpointing: bool = True  # enable transformer.enable_gradient_checkpointing() to fit 1024px
    # bf16 autocast is CUDA-gated (fp32 eager on MPS/CPU for stability), mirroring
    # the wordle samples.
    amp: bool = True
    snapshot_interval: int = 250  # generate sample bufos during training
    checkpoint_interval: int = 500
    eval_interval: int = 0  # >0 runs the cheap CLIP eval every N steps (0 disables)
    num_workers: int = 0  # MPS + dataloader fork issues; keep single-process
    # Prompts used for the periodic in-training preview grid.
    snapshot_prompts: list[str] = [
        "a bufo of happy, frog emoji sticker, white background",
        "a bufo of coffee, frog emoji sticker, white background",
        "a bufo of cowboy, frog emoji sticker, white background",
        "a bufo of crying, frog emoji sticker, white background",
    ]


class BufoLoRAConfig(BaseModel):
    data: DataConfig = DataConfig()
    lora: LoRAConfig = LoRAConfig()
    training: TrainingConfig = TrainingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> BufoLoRAConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))


class EvalConfig(BaseModel):
    """Fixed CLIP-eval settings + the held-out prompt set."""

    concept_text: str = "a bufo, a green cartoon frog sticker"
    # prompt = prompt_template.format(subject=...) + suffix. Both track the caption
    # schema so the benchmark matches each model's training format.
    prompt_template: str = "a bufo of {subject}"
    suffix: str = ", frog emoji sticker, white background"
    # laion ViT-B/32 ships safetensors (openai/clip-vit-base-patch32 is .bin-only,
    # which fails to load on the cluster's torch<2.6 — CVE-2025-32434). Same arch.
    clip_model: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    clipscore_w: float = 2.5
    seed: int = 20260610  # eval seed, decoupled from training seed
    images_per_prompt: int = 4
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = "photo, realistic, 3d render, cluttered, tiny, text, watermark"
    prompts: list[str] = []  # held-out subjects (no suffix)
    step_prompts: list[str] = []  # cheap subset for in-training eval

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
