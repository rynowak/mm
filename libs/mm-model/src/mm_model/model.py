"""Decoder-only GPT model with KV caching and RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from mm_model.config import GPTConfig

KVCache = tuple[Tensor, Tensor]


def _precompute_freqs(head_dim: int, max_len: int, device: torch.device) -> Tensor:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


class MLP(nn.Module):
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
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        freqs: Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        b, t, c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_cache: KVCache = (k, v)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=kv_cache is None,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.out_proj(attn_out)), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        freqs: Tensor,
        kv_cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        attn_out, new_cache = self.attn(self.ln1(x), freqs, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.output_size, bias=config.bias)

        head_dim = config.embed_dim // config.n_heads
        self._rope_freqs: Tensor
        self.register_buffer(
            "_rope_freqs",
            _precompute_freqs(head_dim, config.context_len, torch.device("cpu")),
            persistent=False,
        )

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        kv_cache: list[KVCache] | None = None,
        start_pos: int = 0,
    ) -> tuple[Tensor, Tensor | None, list[KVCache]]:
        b, t = idx.size()

        tok_emb = self.token_emb(idx)
        x = self.drop(tok_emb)

        freqs = self._rope_freqs[start_pos : start_pos + t].to(idx.device)

        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_cache = block(x, freqs, layer_cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        was_training = self.training
        self.eval()

        if use_cache:
            logits, _, kv_cache = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

            for _ in range(max_new_tokens - 1):
                logits, _, kv_cache = self(next_token, kv_cache=kv_cache, start_pos=idx.size(1) - 1)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
        else:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.context_len else idx[:, -self.config.context_len :]
                logits, _, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)

        if was_training:
            self.train()
        return idx
