"""Checkpoint save/load utilities."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from torch.optim import Optimizer

    from mm_model.config import GPTConfig
    from mm_model.model import GPT


def save_checkpoint(path: Path, model: GPT, optimizer: Optimizer, step: int, config: GPTConfig) -> None:
    """Save a training checkpoint.

    Saves model state, optimizer state, training step, config (as dict), and RNG states
    for torch, random, and numpy to enable reproducible resumption.
    """
    from dataclasses import asdict

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": asdict(config),
        "rng_states": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "random": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a training checkpoint.

    Returns a dict with keys: model_state_dict, optimizer_state_dict, step, config, rng_states.
    """
    return torch.load(path, map_location=device, weights_only=False)
