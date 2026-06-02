"""Tests for mm-model: config, model, and checkpointing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mm_model import GPT, GPTConfig, load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

VOCAB_SIZE = 128


def _make_small_config() -> GPTConfig:
    return GPTConfig.small(vocab_size=VOCAB_SIZE)


class TestGPTConfig:
    def test_small_factory(self) -> None:
        cfg = GPTConfig.small(vocab_size=VOCAB_SIZE)
        assert cfg.n_layers == 6
        assert cfg.n_heads == 8
        assert cfg.embed_dim == 256
        assert cfg.vocab_size == VOCAB_SIZE

    def test_medium_factory(self) -> None:
        cfg = GPTConfig.medium(vocab_size=VOCAB_SIZE)
        assert cfg.n_layers == 6
        assert cfg.n_heads == 6
        assert cfg.embed_dim == 384
        assert cfg.vocab_size == VOCAB_SIZE

    def test_small_param_count(self) -> None:
        """Small config should have ~5M params (within 20%)."""
        cfg = GPTConfig.small(vocab_size=VOCAB_SIZE)
        model = GPT(cfg)
        actual = sum(p.numel() for p in model.parameters())
        estimate = cfg.param_count_estimate()
        assert abs(estimate - actual) / actual < 0.20, f"Estimate {estimate} too far from actual {actual}"
        assert 4_000_000 < actual < 6_000_000, f"Small model has {actual} params, expected ~5M"

    def test_medium_param_count(self) -> None:
        """Medium config should have ~10M params (within 20%)."""
        cfg = GPTConfig.medium(vocab_size=VOCAB_SIZE)
        model = GPT(cfg)
        actual = sum(p.numel() for p in model.parameters())
        estimate = cfg.param_count_estimate()
        assert abs(estimate - actual) / actual < 0.20, f"Estimate {estimate} too far from actual {actual}"
        assert 8_000_000 < actual < 12_000_000, f"Medium model has {actual} params, expected ~10M"


class TestGPTModel:
    def test_forward_shape(self) -> None:
        """Forward pass produces logits with correct shape."""
        cfg = _make_small_config()
        model = GPT(cfg)
        model.eval()

        idx = torch.randint(0, VOCAB_SIZE, (2, 16))
        logits, loss, _ = model(idx)

        assert logits.shape == (2, 16, VOCAB_SIZE)
        assert loss is None

    def test_forward_with_targets(self) -> None:
        """Forward pass with targets returns a scalar loss."""
        cfg = _make_small_config()
        model = GPT(cfg)

        idx = torch.randint(0, VOCAB_SIZE, (2, 16))
        targets = torch.randint(0, VOCAB_SIZE, (2, 16))
        logits, loss, _ = model(idx, targets=targets)

        assert logits.shape == (2, 16, VOCAB_SIZE)
        assert loss is not None
        assert loss.ndim == 0  # scalar

    def test_generate_tokens_in_range(self) -> None:
        """Generated tokens should be within vocab range."""
        torch.manual_seed(42)
        cfg = _make_small_config()
        model = GPT(cfg)
        model.eval()

        idx = torch.randint(0, VOCAB_SIZE, (1, 4))
        output = model.generate(idx, max_new_tokens=10)

        assert output.shape == (1, 14)
        assert (output >= 0).all()
        assert (output < VOCAB_SIZE).all()

    def test_generate_with_top_k(self) -> None:
        """Generate with top_k should produce valid tokens."""
        torch.manual_seed(42)
        cfg = _make_small_config()
        model = GPT(cfg)
        model.eval()

        idx = torch.randint(0, VOCAB_SIZE, (1, 4))
        output = model.generate(idx, max_new_tokens=5, temperature=0.8, top_k=10)

        assert output.shape == (1, 9)
        assert (output >= 0).all()
        assert (output < VOCAB_SIZE).all()


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Checkpoint save/load preserves model state, optimizer state, step, and config."""
        torch.manual_seed(42)
        cfg = _make_small_config()
        model = GPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Do a forward+backward to populate optimizer state
        idx = torch.randint(0, VOCAB_SIZE, (2, 8))
        targets = torch.randint(0, VOCAB_SIZE, (2, 8))
        _, loss, _ = model(idx, targets=targets)
        loss.backward()
        optimizer.step()

        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(ckpt_path, model, optimizer, step=42, config=cfg)

        assert ckpt_path.exists()

        loaded = load_checkpoint(ckpt_path, device=torch.device("cpu"))

        assert loaded["step"] == 42
        assert loaded["config"]["n_layers"] == cfg.n_layers
        assert loaded["config"]["embed_dim"] == cfg.embed_dim
        assert loaded["config"]["vocab_size"] == cfg.vocab_size

        # Verify model state can be loaded
        model2 = GPT(cfg)
        model2.load_state_dict(loaded["model_state_dict"])

        # Check weights match
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items(), strict=True):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Mismatch in {k1}"

        # Verify RNG states are present
        assert "rng_states" in loaded
        assert "torch" in loaded["rng_states"]
        assert "random" in loaded["rng_states"]
        assert "numpy" in loaded["rng_states"]
