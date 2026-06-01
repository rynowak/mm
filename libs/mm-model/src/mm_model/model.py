"""Decoder-only GPT model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from mm_model.config import GPTConfig


class MLP(nn.Module):
    """Feed-forward network with 4x expansion and GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.bias)
        self.proj = nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using scaled_dot_product_attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        # Combined QKV projection
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.out_proj(attn_out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual -> LN -> MLP -> residual."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_len, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Forward pass. Returns (logits, loss). Loss is computed if targets are provided."""
        b, t = idx.size()
        assert t <= self.config.context_len, f"Sequence length {t} exceeds context_len {self.config.context_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """Autoregressive generation.

        Args:
            idx: Input token indices of shape (batch, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature. 1.0 = no change, <1.0 = sharper, >1.0 = softer.
            top_k: If set, only sample from the top k most likely tokens.

        Returns:
            Token indices of shape (batch, seq_len + max_new_tokens).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx if idx.size(1) <= self.config.context_len else idx[:, -self.config.context_len :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
