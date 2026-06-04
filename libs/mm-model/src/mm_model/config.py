"""GPT model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for a decoder-only GPT model."""

    n_layers: int
    n_heads: int
    embed_dim: int
    vocab_size: int
    context_len: int = 256
    dropout: float = 0.1
    bias: bool = False
    n_output_classes: int | None = None

    @property
    def output_size(self) -> int:
        return self.n_output_classes if self.n_output_classes is not None else self.vocab_size

    @classmethod
    def small(cls, vocab_size: int, n_output_classes: int | None = None) -> GPTConfig:
        """~5M params: 6 layers, 8 heads, 256 embed dim."""
        return cls(n_layers=6, n_heads=8, embed_dim=256, vocab_size=vocab_size, n_output_classes=n_output_classes)

    @classmethod
    def medium(cls, vocab_size: int, n_output_classes: int | None = None) -> GPTConfig:
        """~10M params: 6 layers, 6 heads, 384 embed dim."""
        return cls(n_layers=6, n_heads=6, embed_dim=384, vocab_size=vocab_size, n_output_classes=n_output_classes)

    def param_count_estimate(self) -> int:
        """Estimate total parameter count for this config.

        Counts embedding, transformer block, and LM head parameters.
        """
        d = self.embed_dim
        v = self.vocab_size
        n = self.n_layers
        bias = 1 if self.bias else 0

        # Token embedding (no positional embedding with RoPE)
        embed_params = v * d

        # Per transformer block:
        #   LayerNorm (pre-attn): 2*d (weight + bias always)
        #   Attention QKV projection: 3 * (d*d + d*bias)
        #   Attention output projection: d*d + d*bias
        #   LayerNorm (pre-MLP): 2*d
        #   MLP fc: d * 4d + 4d*bias
        #   MLP proj: 4d * d + d*bias
        ln_params = 2 * d  # weight + bias
        attn_params = 3 * (d * d + d * bias) + (d * d + d * bias)
        mlp_params = (d * 4 * d + 4 * d * bias) + (4 * d * d + d * bias)
        block_params = 2 * ln_params + attn_params + mlp_params

        # Final LayerNorm
        final_ln_params = 2 * d

        # LM head (weight-tied with token embedding, so no extra params)
        # Actually, we won't tie weights, so count it.
        out_size = self.output_size
        head_params = d * out_size + out_size * bias

        return embed_params + n * block_params + final_ln_params + head_params
