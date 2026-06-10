"""Shared training helpers for V3 phases (pre-train, SFT, RL eval writing)."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import torch
from mm_model import GPT, GPTConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch.utils.data import DataLoader

    from wordle3.config import ModelConfig
    from wordle3.metrics import EvalResult, OpenerMetrics


def build_model(model_cfg: ModelConfig, vocab_size: int) -> GPT:
    cfg = GPTConfig(
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        embed_dim=model_cfg.embed_dim,
        vocab_size=vocab_size,
        context_len=model_cfg.context_len,
        dropout=model_cfg.dropout,
    )
    model = GPT(cfg)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters (vocab {vocab_size})")
    return model


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA; fp32 eager elsewhere (MPS/CPU)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def write_live(run_dir: Path, row: dict, games: list[dict]) -> None:
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "latest.json").write_text(json.dumps({**row, "games": games}))
    with open(live_dir / "history.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def write_snapshot(
    run_dir: Path, step: int, train_res: EvalResult, holdout_res: EvalResult, opener: OpenerMetrics
) -> None:
    snap_dir = run_dir / f"eval-{step}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "snapshot.json").write_text(
        json.dumps(
            {
                # Eval trio (train + hold-out) + avg-guesses-over-wins + opener detail.
                "step": step,
                "win_rate": train_res.win_rate,
                "valid_word_rate": train_res.valid_word_rate,
                "info_gain": train_res.info_gain,
                "avg_guesses": train_res.avg_guesses,  # over wins only (0.0 if none)
                "holdout_win_rate": holdout_res.win_rate,
                "holdout_valid_word_rate": holdout_res.valid_word_rate,
                "holdout_info_gain": holdout_res.info_gain,
                "holdout_avg_guesses": holdout_res.avg_guesses,
                "opener_valid_word_rate": opener.valid_word_rate,
                "opener_info_gain": opener.info_gain,
                "distinct_openers": opener.distinct_valid,
            }
        )
    )
