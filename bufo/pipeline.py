"""Stable Diffusion (1.5 + XL) + FLUX.1 component loading + LoRA helpers.

The model families differ in exactly three places — component loading, per-step
conditioning, and LoRA save/load — so we branch on a ``base_kind`` flag and
isolate the divergence behind ``encode_conditioning`` (keeping the conditioning
call single-path; the train loop branches DDPM-vs-flow-matching once). The base
weights load frozen; only LoRA adapters train.

- sd15/sdxl: DDPM UNet; LoRA on the UNet (always) + text encoder(s) (optional).
- flux: rectified-flow ``FluxTransformer2DModel``; LoRA on the transformer
  attention projections only (both text encoders stay frozen). The "unet" slot
  on ``TrainComponents`` holds the flux transformer so the optimizer/grad-clip/
  save plumbing stays uniform.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

FLUX_DEFAULT_BASE = "black-forest-labs/FLUX.1-dev"


@dataclass
class TrainComponents:
    """Frozen base components plus the (LoRA-trainable) denoiser.

    SDXL adds a second CLIP tokenizer + encoder (``CLIPTextModelWithProjection``).
    FLUX repurposes the fields: ``unet`` holds the ``FluxTransformer2DModel``,
    ``text_encoder`` is CLIP (pooled), ``text_encoder_2`` is the T5 sequence
    encoder, ``tokenizer_2`` is the T5 tokenizer, and ``noise_scheduler`` is unused
    (flow matching needs no DDPM scheduler — see train_lora.flux_flow_match_loss).
    """

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: Any  # UNet2DConditionModel (sd15/sdxl) or FluxTransformer2DModel (flux)
    noise_scheduler: DDPMScheduler | None
    base_kind: str = "sd15"
    tokenizer_2: CLIPTokenizer | None = None
    text_encoder_2: Any | None = None  # CLIPTextModelWithProjection (sdxl) | T5EncoderModel (flux)


def _load_flux_components(base_model: str, device: torch.device, dtype: torch.dtype) -> TrainComponents:
    """Load FLUX.1 parts (transformer/VAE/CLIP/T5); freeze all base weights.

    Mirrors the component set in diffusers' train_dreambooth_lora_flux.py: a CLIP
    text encoder (pooled, dim 768), a T5 sequence encoder (dim 4096), the 16-channel
    AutoencoderKL (uses both shift_factor and scaling_factor), and the rectified-flow
    FluxTransformer2DModel. No DDPM scheduler — flow matching is scheduler-free here.
    """
    from diffusers import FluxTransformer2DModel
    from transformers import AutoTokenizer, T5EncoderModel

    # Host RAM on the GPU worker pod is capped (~12GB). Loading the 12B transformer /
    # 4.7B T5 in fp32 on CPU (the default) OOM-kills the worker, so the big models are
    # loaded in `dtype` (bf16) and STREAMED straight to the GPU via device_map (host
    # never holds the full fp32 copy) — the same pattern the Qwen judge loader uses.
    # device_map={"": device} pins everything to one GPU (no offload hooks → trainable).
    # safetensors needs an *indexed* device ("cuda:0"); a bare torch.device("cuda")
    # is rejected ("device cuda is invalid"), so pass the GPU index for device_map.
    dev_target = 0 if device.type == "cuda" else str(device)
    big = {"torch_dtype": dtype, "device_map": {"": dev_target}, "low_cpu_mem_usage": True}

    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    # The flux tokenizer_2 subfolder is a T5TokenizerFast; AutoTokenizer resolves it
    # from the saved tokenizer config (robust across transformers 4.x/5.x, where the
    # explicit T5TokenizerFast symbol is an alias not present in all type stubs).
    tokenizer_2 = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")
    # Move CLIP + VAE to the GPU *immediately* so host RAM is clear before the big
    # device_map streams. The pod's ~12GB host cap is tight enough that Flux barely
    # fits; leaving even ~0.5GB of encoders resident on host during the transformer
    # stream is enough to OOM the worker.
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
    text_encoder_2 = T5EncoderModel.from_pretrained(base_model, subfolder="text_encoder_2", **big)
    transformer = FluxTransformer2DModel.from_pretrained(base_model, subfolder="transformer", **big)

    for module in (vae, transformer, text_encoder, text_encoder_2):
        module.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    # transformer + text_encoder_2 are already on-device via device_map; don't .to() them
    # (accelerate-loaded models reject .to). They're frozen; the LoRA adapters add later.
    text_encoder_2.eval()

    return TrainComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=transformer,
        noise_scheduler=None,
        base_kind="flux",
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
    )


def load_train_components(
    base_model: str, device: torch.device, *, base_kind: str = "sd15", dtype: torch.dtype = torch.float32
) -> TrainComponents:
    """Load tokenizer/text-encoder(s)/VAE/denoiser/scheduler; freeze all base weights."""
    if base_kind == "flux":
        return _load_flux_components(base_model, device, dtype)

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
    """Attach LoRA to the denoiser (always) and text encoder(s) (sd15/sdxl, if requested).

    Returns the list of modules with trainable params (for the optimizer + grad
    clip). Adapter params are forced to fp32 even when the base is lower precision.

    For flux the denoiser is the FluxTransformer2DModel and we adapt its attention
    projections (cfg.flux_target_modules); both text encoders stay frozen, matching
    diffusers' train_dreambooth_lora_flux.py default (no --train_text_encoder there).
    """
    if comp.base_kind == "flux":
        transformer_lora = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=cfg.flux_target_modules,
            init_lora_weights="gaussian",
        )
        comp.unet.add_adapter(transformer_lora)
        trained_flux: list[torch.nn.Module] = [comp.unet]
        for param in comp.unet.parameters():
            if param.requires_grad:
                param.data = param.data.float()
        return trained_flux

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
    """Write ``pytorch_lora_weights.safetensors`` (denoiser + any text-encoder layers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_layers = _adapter_state(comp.unet)
    if comp.base_kind == "flux":
        from diffusers import FluxPipeline

        FluxPipeline.save_lora_weights(
            str(out_dir),
            transformer_lora_layers=unet_layers,
            safe_serialization=True,
        )
        return
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
    # weights_only=False: our own checkpoint carries optimizer/scheduler/RNG state
    # (non-tensor objects), which the torch>=2.6 safe loader rejects.
    state = torch.load(path, map_location="cpu", weights_only=False)
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
    """Denoiser conditioning kwargs for one batch (model-kind aware).

    Not wrapped in ``no_grad`` so text-encoder LoRA gradients can flow; a frozen
    encoder simply produces no graph.

    For flux returns the raw flow-matching conditioning (not transformer kwargs):
    ``pooled_projections`` (CLIP pooler_output, dim 768), ``encoder_hidden_states``
    (T5 last_hidden_state, dim 4096), and ``txt_ids`` (zeros [seq, 3]). The flux
    train path assembles the transformer call. Mirrors _encode_prompt_with_clip /
    _encode_prompt_with_t5 / encode_prompt in train_dreambooth_lora_flux.py.
    """
    if comp.base_kind == "flux":
        # CLIP -> pooled (no sequence used by flux). pooler_output: [B, 768].
        pooled = comp.text_encoder(batch["input_ids"].to(device), output_hidden_states=False).pooler_output
        # T5 -> sequence embeds. last_hidden_state: [B, seq, 4096].
        prompt_embeds = comp.text_encoder_2(batch["input_ids_2"].to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        pooled = pooled.to(dtype=dtype)
        # text_ids: positional ids for the text stream, all zeros (single image, no
        # T5 RoPE offset). Shape [seq, 3]; shared across the batch.
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
        return {
            "pooled_projections": pooled,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
        }

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
    sampler: str | None = None,
    lora_scale: float = 1.0,
) -> Any:
    """Build an SD/SDXL/FLUX pipeline (safety checker disabled for SD1.5) + optional LoRA.

    ``sampler`` (sd15/sdxl only) overrides the base's default scheduler. SDXL defaults to
    Euler, which is the most artifact-prone; "dpmpp_2m_karras" (DPM++ 2M Karras) is the
    cleaner choice. Flux ignores it (flow-matching has its own scheduler).
    """
    if base_kind == "flux":
        from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline

        # The GPU worker pod caps host RAM at ~12GB. FluxPipeline.from_pretrained
        # re-materializes the 12B transformer on the host and OOM-kills the worker, so
        # reuse the proven streaming loader (_load_flux_components: device_map → GPU) and
        # assemble the pipeline from those live components via the CONSTRUCTOR — exactly
        # what build_live_pipe does for the in-training snapshots (which work).
        flux_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        comp = _load_flux_components(base_model, device, flux_dtype)
        flux_pipe = FluxPipeline(
            vae=comp.vae,
            text_encoder=comp.text_encoder,
            text_encoder_2=comp.text_encoder_2,
            tokenizer=comp.tokenizer,
            tokenizer_2=comp.tokenizer_2,
            transformer=comp.unet,
            # Flux's OWN scheduler config (shift + use_dynamic_shifting) — a bare
            # FlowMatchEulerDiscreteScheduler() has no shift and under-denoises → blurry.
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler"),
        )
        if lora_dir is not None:
            # load_lora_weights routes transformer_lora_layers into the transformer (tiny, 26M).
            flux_pipe.load_lora_weights(str(lora_dir))
            if lora_scale != 1.0:
                flux_pipe.fuse_lora(lora_scale=lora_scale)
        return flux_pipe
    if base_kind == "sdxl":
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=dtype
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
    pipe.to(device)
    if sampler:
        from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

        if sampler in ("dpmpp_2m_karras", "dpmpp_2m"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=sampler.endswith("karras")
            )
        elif sampler in ("dpmpp_sde_karras", "dpmpp_sde"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=sampler.endswith("karras")
            )
        elif sampler == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            raise ValueError(f"Unknown sampler '{sampler}' (use dpmpp_2m_karras|dpmpp_sde_karras|euler)")
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir))
        # A high-rank LoRA at full strength over-applies the trigger (subject tiling /
        # duplication). fuse_lora(lora_scale=) bakes the adapter in at a chosen strength so
        # eval can read the model at the strength it'll actually be used at (~0.7), not 1.0.
        if lora_scale != 1.0:
            pipe.fuse_lora(lora_scale=lora_scale)
    return pipe


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA; fp32 eager elsewhere (MPS/CPU), mirroring the wordle samples."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
