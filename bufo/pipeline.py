"""Stable Diffusion (1.5 + XL) component loading + LoRA helpers.

The two model families differ in exactly three places — component loading,
per-step conditioning, and LoRA save/load — so we branch on a ``base_kind`` flag
and isolate the divergence behind ``encode_conditioning`` (keeping the training
loop single-path). The base weights load frozen; only LoRA adapters train (UNet
always, plus the text encoder(s) when ``train_text_encoder``).
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from bufo.config import LoRAConfig as BufoLoRAConfig


@dataclass
class TrainComponents:
    """Frozen SD components plus the (LoRA-trainable) UNet. SDXL adds a second
    tokenizer + text encoder (``CLIPTextModelWithProjection``)."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler
    base_kind: str = "sd15"
    tokenizer_2: CLIPTokenizer | None = None
    text_encoder_2: CLIPTextModelWithProjection | None = None


def load_train_components(
    base_model: str, device: torch.device, *, base_kind: str = "sd15", dtype: torch.dtype = torch.float32
) -> TrainComponents:
    """Load tokenizer/text-encoder(s)/VAE/UNet/scheduler; freeze all base weights."""
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    tokenizer_2 = text_encoder_2 = None
    if base_kind == "sdxl":
        tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2")

    encoders = [text_encoder] + ([text_encoder_2] if text_encoder_2 is not None else [])
    for module in (vae, unet, *encoders):
        module.requires_grad_(False)
    vae.to(device, dtype=dtype).eval()
    unet.to(device, dtype=dtype)
    for enc in encoders:
        enc.to(device, dtype=dtype).eval()

    return TrainComponents(tokenizer, text_encoder, vae, unet, noise_scheduler, base_kind, tokenizer_2, text_encoder_2)


def attach_lora(comp: TrainComponents, cfg: BufoLoRAConfig) -> list[torch.nn.Module]:
    """Attach LoRA to the UNet (always) and text encoder(s) (if requested).

    Returns the list of modules with trainable params (for the optimizer + grad
    clip). Adapter params are forced to fp32 even when the base is lower precision.
    """
    unet_lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        init_lora_weights="gaussian",
    )
    comp.unet.add_adapter(unet_lora)
    trained: list[torch.nn.Module] = [comp.unet]

    if cfg.train_text_encoder:
        te_lora = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.text_target_modules,
            init_lora_weights="gaussian",
        )
        for enc in (comp.text_encoder, comp.text_encoder_2):
            if enc is not None:
                enc.add_adapter(te_lora)
                trained.append(enc)

    for module in trained:
        for param in module.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    return trained


def _adapter_state(module: torch.nn.Module) -> dict | None:
    if getattr(module, "peft_config", None):
        return convert_state_dict_to_diffusers(get_peft_model_state_dict(module))
    return None


def save_lora(comp: TrainComponents, out_dir: Path) -> None:
    """Write ``pytorch_lora_weights.safetensors`` (UNet + any text-encoder layers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_layers = _adapter_state(comp.unet)
    if comp.base_kind == "sdxl":
        StableDiffusionXLPipeline.save_lora_weights(
            str(out_dir),
            unet_lora_layers=unet_layers,
            text_encoder_lora_layers=_adapter_state(comp.text_encoder),
            text_encoder_2_lora_layers=_adapter_state(comp.text_encoder_2) if comp.text_encoder_2 else None,
            safe_serialization=True,
        )
    else:
        StableDiffusionPipeline.save_lora_weights(
            str(out_dir),
            unet_lora_layers=unet_layers,
            text_encoder_lora_layers=_adapter_state(comp.text_encoder),
            safe_serialization=True,
        )


def _adapter_modules(comp: TrainComponents) -> dict[str, torch.nn.Module]:
    """Trainable LoRA-bearing modules keyed by a stable name (for resume)."""
    mods: dict[str, torch.nn.Module] = {"unet_lora": comp.unet}
    if getattr(comp.text_encoder, "peft_config", None):
        mods["text_encoder_lora"] = comp.text_encoder
    if comp.text_encoder_2 is not None and getattr(comp.text_encoder_2, "peft_config", None):
        mods["text_encoder_2_lora"] = comp.text_encoder_2
    return mods


def save_training_state(
    comp: TrainComponents, optimizer: Optimizer, scheduler: LRScheduler, step: int, out_dir: Path
) -> None:
    """Save a full resumable checkpoint: adapters + optimizer + scheduler + step + RNG.

    Sits beside the inference ``pytorch_lora_weights.safetensors`` in the same dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state: dict = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    for name, module in _adapter_modules(comp).items():
        state[name] = get_peft_model_state_dict(module)
    torch.save(state, out_dir / "training_state.pt")


def load_training_state(
    path: Path, comp: TrainComponents, optimizer: Optimizer, scheduler: LRScheduler, device: torch.device
) -> int:
    """Restore a checkpoint saved by :func:`save_training_state`. Returns the step.

    Requires the same LoRA config (adapters must already be attached). Optimizer
    state is moved onto ``device`` since it loads to CPU.
    """
    state = torch.load(path, map_location="cpu")
    for name, module in _adapter_modules(comp).items():
        set_peft_model_state_dict(module, state[name])
    optimizer.load_state_dict(state["optimizer"])
    for opt_state in optimizer.state.values():
        for key, value in opt_state.items():
            if isinstance(value, torch.Tensor):
                opt_state[key] = value.to(device)
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_torch"])
    np.random.set_state(state["rng_numpy"])
    random.setstate(state["rng_python"])
    return int(state["step"])


def encode_conditioning(
    comp: TrainComponents, batch: dict, *, resolution: int, device: torch.device, dtype: torch.dtype
) -> dict:
    """UNet conditioning kwargs for one batch (model-kind aware).

    Not wrapped in ``no_grad`` so text-encoder LoRA gradients can flow; a frozen
    encoder simply produces no graph.
    """
    if comp.base_kind == "sd15":
        return {"encoder_hidden_states": comp.text_encoder(batch["input_ids"].to(device))[0]}

    ids_1, ids_2 = batch["input_ids"].to(device), batch["input_ids_2"].to(device)
    out_1 = comp.text_encoder(ids_1, output_hidden_states=True)
    out_2 = comp.text_encoder_2(ids_2, output_hidden_states=True)
    # SDXL concatenates the PENULTIMATE hidden states of both encoders (768+1280),
    # with pooled embeds from encoder 2 only.
    prompt_embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
    pooled = out_2[0]
    # Constant micro-conditioning: our images are pre-squared so original == target
    # size and crop offset is (0, 0).
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]], dtype=dtype, device=device
    ).repeat(ids_1.shape[0], 1)
    return {
        "encoder_hidden_states": prompt_embeds,
        "added_cond_kwargs": {"text_embeds": pooled, "time_ids": add_time_ids},
    }


def load_inference_pipeline(
    base_model: str,
    device: torch.device,
    lora_dir: Path | None = None,
    *,
    base_kind: str = "sd15",
    dtype: torch.dtype = torch.float32,
) -> StableDiffusionPipeline | StableDiffusionXLPipeline:
    """Build an SD/SDXL pipeline (safety checker disabled for SD1.5) + optional LoRA."""
    if base_kind == "sdxl":
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=dtype
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
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
