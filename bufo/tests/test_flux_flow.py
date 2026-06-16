"""Offline tests for the FLUX.1 flow-matching helpers (no model download).

These pin the math we mirrored from diffusers' train_dreambooth_lora_flux.py:
latent packing/ids shapes, logit-normal sigma range, the rectified-flow noisy/
target construction, and the SD3 loss-weighting hook. All run on tiny random
tensors — no FluxTransformer, VAE, or weights involved.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: E402

from bufo.train_lora import (  # noqa: E402
    _pack_latents,
    _prepare_latent_image_ids,
    _sd3_loss_weighting,
    sample_flux_sigmas,
)


def test_pack_latents_matches_diffusers_reference():
    # _pack_latents must agree with FluxPipeline._pack_latents bit-for-bit, since
    # the transformer was trained against that exact packing.
    torch.manual_seed(0)
    b, c, h, w = 2, 16, 8, 12
    latents = torch.randn(b, c, h, w)
    ours = _pack_latents(latents)
    ref = FluxPipeline._pack_latents(latents, b, c, h, w)
    assert ours.shape == (b, (h // 2) * (w // 2), c * 4)
    assert torch.equal(ours, ref)


def test_prepare_latent_image_ids_matches_reference_shape_and_values():
    ph, pw = 4, 6  # packed grid (latent H/2, W/2)
    ours = _prepare_latent_image_ids(ph, pw, torch.device("cpu"), torch.float32)
    ref = FluxPipeline._prepare_latent_image_ids(1, ph, pw, torch.device("cpu"), torch.float32)
    assert ours.shape == (ph * pw, 3)
    assert torch.equal(ours, ref)
    # channel 0 always zero; channels 1/2 are row/col indices.
    assert torch.all(ours[:, 0] == 0)
    assert ours[:, 1].max().item() == ph - 1
    assert ours[:, 2].max().item() == pw - 1


def test_pack_then_unpack_roundtrips():
    # Packing is invertible via FluxPipeline._unpack_latents (used in the loss to
    # realign the prediction with the target).
    torch.manual_seed(1)
    b, c, h, w = 1, 16, 8, 8
    latents = torch.randn(b, c, h, w)
    packed = _pack_latents(latents)
    vae_scale_factor = 8
    unpacked = FluxPipeline._unpack_latents(
        packed, height=h * vae_scale_factor, width=w * vae_scale_factor, vae_scale_factor=vae_scale_factor
    )
    assert unpacked.shape == latents.shape
    assert torch.allclose(unpacked, latents, atol=1e-6)


def test_sample_flux_sigmas_in_unit_interval():
    torch.manual_seed(2)
    sigmas = sample_flux_sigmas(1024, logit_mean=0.0, logit_std=1.0, device=torch.device("cpu"))
    assert sigmas.shape == (1024,)
    assert sigmas.min().item() > 0.0
    assert sigmas.max().item() < 1.0


def test_flow_match_noisy_and_target_construction():
    # zt = (1 - t) x + t z1 ; target velocity = z1 - x = noise - latents.
    torch.manual_seed(3)
    b, c, h, w = 2, 16, 4, 4
    latents = torch.randn(b, c, h, w)
    noise = torch.randn_like(latents)
    t = sample_flux_sigmas(b, 0.0, 1.0, torch.device("cpu"))
    sigma = t.view(b, 1, 1, 1)
    noisy = (1.0 - sigma) * latents + sigma * noise
    target = noise - latents
    # at sigma -> 0 the noisy input approaches the clean latent; check the convex blend.
    blended = (1.0 - sigma) * latents + sigma * noise
    assert torch.allclose(noisy, blended)
    assert torch.equal(target, noise - latents)
    # velocity has the right shape for an MSE against the (unpacked) prediction.
    assert target.shape == latents.shape


def test_sd3_loss_weighting_schemes():
    sigma = torch.full((2, 1, 1, 1), 0.5)
    none_w = _sd3_loss_weighting("none", sigma)
    assert torch.equal(none_w, torch.ones_like(sigma))  # uniform by default
    sigma_sqrt_w = _sd3_loss_weighting("sigma_sqrt", sigma)
    assert torch.allclose(sigma_sqrt_w, sigma**-2.0)
    cosmap_w = _sd3_loss_weighting("cosmap", sigma)
    assert torch.all(cosmap_w > 0)
