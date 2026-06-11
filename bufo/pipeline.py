"""Stable Diffusion component loading + LoRA attach/save/load helpers.

Shared by training (`train_lora`) and inference (`sample`). The base model is
loaded frozen; only low-rank adapters on the UNet attention projections train.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from bufo.config import LoRAConfig as BufoLoRAConfig


@dataclass
class TrainComponents:
    """Frozen SD components plus the (LoRA-trainable) UNet."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler


def load_train_components(base_model: str, device: torch.device, dtype: torch.dtype = torch.float32) -> TrainComponents:
    """Load tokenizer/text-encoder/VAE/UNet/scheduler; freeze all base weights."""
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(device, dtype=dtype).eval()
    text_encoder.to(device, dtype=dtype).eval()
    unet.to(device, dtype=dtype)

    return TrainComponents(tokenizer, text_encoder, vae, unet, noise_scheduler)


def attach_lora(unet: UNet2DConditionModel, cfg: BufoLoRAConfig) -> None:
    """Add LoRA adapters to the UNet attention projections (in place).

    Adapter params are forced to fp32 even when the base is lower precision —
    optimizer states stay stable and the tiny extra memory is negligible.
    """
    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora)
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.float()


def trainable_params(unet: UNet2DConditionModel) -> list[torch.nn.Parameter]:
    return [p for p in unet.parameters() if p.requires_grad]


def save_lora(unet: UNet2DConditionModel, out_dir: Path) -> None:
    """Write ``pytorch_lora_weights.safetensors`` loadable via ``load_lora_weights``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(str(out_dir), unet_lora_layers=state_dict, safe_serialization=True)


def load_inference_pipeline(
    base_model: str,
    device: torch.device,
    lora_dir: Path | None = None,
    dtype: torch.dtype = torch.float32,
) -> StableDiffusionPipeline:
    """Build an SD pipeline (safety checker disabled) with optional LoRA weights."""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA; fp32 eager elsewhere (MPS/CPU), mirroring the wordle samples."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
