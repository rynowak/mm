"""Tests for the diffusers->kohya LoRA key conversion."""

from __future__ import annotations

import torch

from bufo.lora_to_kohya import convert


def _fake_diffusers_lora(rank: int = 4) -> dict[str, torch.Tensor]:
    """A minimal diffusers/PEFT-style SDXL LoRA state dict (unet + both text encoders)."""
    sd: dict[str, torch.Tensor] = {}

    def pair(module: str, lora_suffix: str) -> None:
        sd[f"{module}.{lora_suffix}.down.weight"] = torch.randn(rank, 8)
        sd[f"{module}.{lora_suffix}.up.weight"] = torch.randn(8, rank)

    attn = "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1"
    pair(f"{attn}.to_k", "lora")
    pair(f"{attn}.to_out.0", "lora")  # the to_out.0 form
    te = "text_model.encoder.layers.0.self_attn"
    pair(f"text_encoder.{te}.k_proj", "lora_linear_layer")
    pair(f"text_encoder_2.{te}.q_proj", "lora_linear_layer")
    return sd


def test_convert_key_renaming() -> None:
    ko = convert(_fake_diffusers_lora(), fp16=False)
    # prefixes mapped
    assert any(k.startswith("lora_unet_") for k in ko)
    assert any(k.startswith("lora_te1_") for k in ko)
    assert any(k.startswith("lora_te2_") for k in ko)
    # no diffusers-style keys leak through
    assert not any(k.startswith(("unet.", "text_encoder")) for k in ko)
    # to_out.0 -> to_out_0, and lora.down -> lora_down
    assert "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight" in ko
    # te uses lora_te1_ + the text_model path
    assert "lora_te1_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight" in ko


def test_convert_synthesizes_alpha_equal_to_rank() -> None:
    rank = 7
    ko = convert(_fake_diffusers_lora(rank=rank), fp16=False)
    alphas = {k: v for k, v in ko.items() if k.endswith(".alpha")}
    # one alpha per module (4 modules: 2 unet + 2 te)
    assert len(alphas) == 4
    for v in alphas.values():
        assert float(v) == float(rank)


def test_fp16_cast() -> None:
    ko = convert(_fake_diffusers_lora(), fp16=True)
    weights = [v for k, v in ko.items() if k.endswith(".weight")]
    assert weights and all(v.dtype == torch.float16 for v in weights)
