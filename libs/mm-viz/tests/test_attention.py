"""Tests for attention weight extraction and visualization."""

from __future__ import annotations

import torch
from mm_model.config import GPTConfig
from mm_model.model import GPT
from mm_viz.attention import (
    extract_attention_weights,
    render_attention_html,
    render_wordle_attention_html,
)


def _make_small_model() -> GPT:
    """Create a tiny GPT for testing (2 layers, 2 heads, 32 embed_dim)."""
    config = GPTConfig(
        n_layers=2,
        n_heads=2,
        embed_dim=32,
        vocab_size=50,
        context_len=64,
        dropout=0.0,
    )
    return GPT(config)


class TestExtractAttentionWeights:
    def test_returns_correct_number_of_layers(self) -> None:
        model = _make_small_model()
        input_ids = torch.randint(0, 50, (1, 8))
        weights = extract_attention_weights(model, input_ids, torch.device("cpu"))
        assert len(weights) == 2

    def test_correct_shapes(self) -> None:
        model = _make_small_model()
        seq_len = 10
        input_ids = torch.randint(0, 50, (1, seq_len))
        weights = extract_attention_weights(model, input_ids, torch.device("cpu"))
        for w in weights:
            assert w.shape == (2, seq_len, seq_len)

    def test_weights_sum_to_one(self) -> None:
        model = _make_small_model()
        input_ids = torch.randint(0, 50, (1, 6))
        weights = extract_attention_weights(model, input_ids, torch.device("cpu"))
        for w in weights:
            # Each row should sum to ~1.0 (softmax output)
            row_sums = w.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_applied(self) -> None:
        model = _make_small_model()
        input_ids = torch.randint(0, 50, (1, 6))
        weights = extract_attention_weights(model, input_ids, torch.device("cpu"))
        for w in weights:
            # Upper triangle (above diagonal) should be zero (causal mask)
            for i in range(w.shape[1]):
                for j in range(i + 1, w.shape[2]):
                    assert w[:, i, j].sum().item() == 0.0

    def test_hooks_removed_after_extraction(self) -> None:
        model = _make_small_model()
        input_ids = torch.randint(0, 50, (1, 5))

        # Count hooks before
        hooks_before = sum(len(m._forward_hooks) for m in model.modules())

        extract_attention_weights(model, input_ids, torch.device("cpu"))

        # Count hooks after — should be same as before
        hooks_after = sum(len(m._forward_hooks) for m in model.modules())
        assert hooks_after == hooks_before


class TestRenderAttentionHtml:
    def _make_weights(self) -> list[torch.Tensor]:
        """Create dummy attention weights for 2 layers, 2 heads, 4 tokens."""
        return [torch.rand(2, 4, 4).softmax(dim=-1) for _ in range(2)]

    def test_produces_html_with_styles(self) -> None:
        weights = self._make_weights()
        tokens = ["a", "b", "c", "d"]
        html = render_attention_html(weights, tokens)
        assert "rgb(" in html
        assert "<table" in html

    def test_contains_title(self) -> None:
        weights = self._make_weights()
        tokens = ["a", "b", "c", "d"]
        html = render_attention_html(weights, tokens, title="Test Title")
        assert "Test Title" in html

    def test_specific_layer_and_head(self) -> None:
        weights = self._make_weights()
        tokens = ["a", "b", "c", "d"]
        html = render_attention_html(weights, tokens, layer=0, head=1)
        assert "Layer 0, Head 1" in html
        # Should not contain Layer 1
        assert "Layer 1" not in html

    def test_specific_layer_all_heads(self) -> None:
        weights = self._make_weights()
        tokens = ["a", "b", "c", "d"]
        html = render_attention_html(weights, tokens, layer=0)
        assert "Layer 0, Head 0" in html
        assert "Layer 0, Head 1" in html
        # Should not contain Layer 1
        assert "Layer 1" not in html

    def test_all_layers_mean(self) -> None:
        weights = self._make_weights()
        tokens = ["a", "b", "c", "d"]
        html = render_attention_html(weights, tokens)
        assert "Layer 0 (mean)" in html
        assert "Layer 1 (mean)" in html

    def test_token_labels_in_output(self) -> None:
        weights = self._make_weights()
        tokens = ["hello", "world", "foo", "bar"]
        html = render_attention_html(weights, tokens)
        for tok in tokens:
            assert tok in html


class TestRenderWordleAttentionHtml:
    def _make_weights(self) -> list[torch.Tensor]:
        """Create dummy attention weights for 2 layers, 2 heads, 6 tokens."""
        return [torch.rand(2, 6, 6).softmax(dim=-1) for _ in range(2)]

    def test_produces_html(self) -> None:
        weights = self._make_weights()
        tokens = ["s", "l", "a", "t", "e", "\n"]
        html = render_wordle_attention_html(weights, tokens)
        assert "<div" in html
        assert "Wordle Attention Patterns" in html

    def test_custom_title(self) -> None:
        weights = self._make_weights()
        tokens = ["s", "l", "a", "t", "e", "\n"]
        html = render_wordle_attention_html(weights, tokens, title="Custom")
        assert "Custom" in html

    def test_last_position_section(self) -> None:
        weights = self._make_weights()
        tokens = ["s", "l", "a", "t", "e", "\n"]
        html = render_wordle_attention_html(weights, tokens)
        assert "Last-Position Attention" in html
        assert "Next Token Prediction" in html

    def test_full_heatmap_section(self) -> None:
        weights = self._make_weights()
        tokens = ["s", "l", "a", "t", "e", "\n"]
        html = render_wordle_attention_html(weights, tokens)
        assert "Full Attention Heatmaps" in html
        assert "Layer 0 (mean)" in html
