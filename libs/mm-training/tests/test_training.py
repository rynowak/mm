"""Tests for mm-training library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import pathlib

from mm_training.device import get_device
from mm_training.manifest import RunManifest
from mm_training.metrics import MetricsLogger
from mm_training.optim import create_optimizer, create_scheduler
from mm_training.utils import clip_grad_norm, seed_everything


def _tiny_model() -> torch.nn.Module:
    """Create a minimal model with params that exercise decay/no-decay splitting."""
    model = torch.nn.Sequential(
        torch.nn.Embedding(10, 8),
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4),
    )
    return model


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_get_device_returns_valid_device() -> None:
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_create_optimizer_two_param_groups() -> None:
    model = _tiny_model()
    optimizer = create_optimizer(model)

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Both groups should have parameters
    assert len(decay_group["params"]) > 0
    assert len(no_decay_group["params"]) > 0

    # Total params should equal the model's trainable params
    total_optim_params = len(decay_group["params"]) + len(no_decay_group["params"])
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_optim_params == total_model_params


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_create_scheduler_warmup_curve_shape() -> None:
    model = _tiny_model()
    optimizer = create_optimizer(model, lr=1.0)
    warmup_steps = 10
    total_steps = 100
    scheduler = create_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    lrs: list[float] = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # LR starts low (near 0)
    assert lrs[0] < 0.05

    # LR peaks at end of warmup
    peak_lr = lrs[warmup_steps - 1]
    assert peak_lr > 0.8

    # LR decays after warmup
    assert lrs[-1] < peak_lr

    # LR at end should be near 10% of peak
    assert lrs[-1] < 0.2


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


def test_metrics_logger_creates_dir(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "runs" / "test_exp" / "20240101_000000"
    logger = MetricsLogger("test_exp", run_dir=run_dir)

    assert logger.log_dir == run_dir
    assert run_dir.is_dir()

    logger.close()


def test_metrics_logger_logs_without_error(tmp_path: pathlib.Path) -> None:
    logger = MetricsLogger("test_exp", run_dir=tmp_path / "logs")

    logger.log_scalar("loss", 0.5, step=1)
    logger.log_scalars("metrics", {"acc": 0.9, "loss": 0.1}, step=1)
    logger.log_histogram("weights", torch.randn(100), step=1)
    logger.log_text("info", "hello", step=1)

    logger.close()


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def test_clip_grad_norm_returns_norm_and_clips() -> None:
    model = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()

    max_norm = 0.1
    pre_clip_norm = clip_grad_norm(model, max_norm=max_norm)

    # Pre-clip norm should be positive
    assert pre_clip_norm > 0

    # After clipping, actual norm should be <= max_norm (with tolerance)
    post_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
    assert post_clip_norm <= max_norm + 1e-6


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------


def test_seed_everything_deterministic() -> None:
    seed_everything(42)
    a = torch.randn(5)

    seed_everything(42)
    b = torch.randn(5)

    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------


def test_run_manifest_capture_and_roundtrip(tmp_path: pathlib.Path) -> None:
    manifest = RunManifest.capture(
        experiment="test_exp",
        config={"lr": 3e-4, "batch_size": 32},
        seed=42,
        dataset_id="test_dataset",
        dataset_revision="abc123",
    )

    assert manifest.experiment == "test_exp"
    assert manifest.seed == 42
    assert "python" in manifest.package_versions
    assert "torch" in manifest.package_versions
    assert "device_type" in manifest.hardware

    # Save and load round-trip
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert path.exists()

    loaded = RunManifest.load(path)
    assert loaded.experiment == manifest.experiment
    assert loaded.config == manifest.config
    assert loaded.seed == manifest.seed
    assert loaded.dataset_id == manifest.dataset_id
    assert loaded.dataset_revision == manifest.dataset_revision
    assert loaded.package_versions == manifest.package_versions
    assert loaded.git_commit == manifest.git_commit
    assert loaded.hardware == manifest.hardware
