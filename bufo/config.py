"""Pydantic config models for the bufo LoRA fine-tuning sample."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 16  # scaling = alpha / rank
    dropout: float = 0.0
    # SD UNet cross/self-attention projection names (peft matches by suffix).
    target_modules: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]


class TrainingConfig(BaseModel):
    seed: int = 42
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    batch_size: int = 1
    grad_accum: int = 4  # effective batch = batch_size * grad_accum
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 50
    max_steps: int = 1500
    grad_clip: float = 1.0
    # bf16 autocast is CUDA-gated (fp32 eager on MPS/CPU for stability), mirroring
    # the wordle samples.
    amp: bool = True
    snapshot_interval: int = 250  # generate sample bufos during training
    checkpoint_interval: int = 500
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
