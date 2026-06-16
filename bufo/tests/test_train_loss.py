"""Offline tests for the diffusion loss (incl. min-SNR weighting)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("diffusers")
from diffusers import DDPMScheduler  # noqa: E402

from bufo.train_lora import diffusion_loss  # noqa: E402


def _comp(prediction_type: str = "epsilon"):
    sched = DDPMScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    return SimpleNamespace(noise_scheduler=sched)


def _batch():
    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8), torch.tensor([10, 500])


def test_gamma_zero_is_plain_mse():
    pred, target, ts = _batch()
    assert torch.allclose(diffusion_loss(pred, target, ts, _comp(), 0.0), F.mse_loss(pred, target))


def test_min_snr_is_positive_and_reweights():
    pred, target, ts = _batch()
    weighted = diffusion_loss(pred, target, ts, _comp(), 5.0)
    assert weighted.item() > 0
    assert not torch.allclose(weighted, F.mse_loss(pred, target))  # weighting changed it


def test_min_snr_v_prediction_runs():
    pred, target, ts = _batch()
    assert diffusion_loss(pred, target, ts, _comp("v_prediction"), 5.0).item() > 0
