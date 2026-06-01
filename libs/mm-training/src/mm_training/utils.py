"""Training utilities: gradient clipping and RNG seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def clip_grad_norm(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm and return the pre-clip norm."""
    total_norm: float = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    ).item()
    return total_norm


def seed_everything(seed: int) -> None:
    """Seed torch, numpy, and python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
