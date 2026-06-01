"""Optimizer and learning-rate schedule construction."""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """Create AdamW with two parameter groups: decay and no-decay.

    No-decay group: biases, LayerNorm weights/biases, and embedding weights.
    Decay group: everything else (linear weights).
    """
    no_decay_names = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    no_decay_types = (torch.nn.LayerNorm, torch.nn.Embedding)

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check by name suffix (handles dotted paths like "attn.bias")
        if any(name.endswith(nd) for nd in no_decay_names):
            no_decay_params.append(param)
            continue

        # Check by parent module type
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules()).get(parts[0])
            if isinstance(parent, no_decay_types):
                no_decay_params.append(param)
                continue

        decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas)


def create_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LRScheduler:
    """Linear warmup then cosine decay to 10% of max LR.

    Uses LambdaLR with a custom lr_lambda function.
    """
    min_lr_ratio = 0.1

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: 0 -> 1 over warmup_steps
            return current_step / max(1, warmup_steps)
        # Cosine decay: 1 -> min_lr_ratio over remaining steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
