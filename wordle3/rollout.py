"""Batched group rollout for RL (§5.7-B).

The whole group shares one prompt and decodes 5 letters in lockstep, so sampling
is one batched, KV-cached decode instead of ``group_size`` independent rollouts —
the dominant RL speedup. Log-probs are a single batched forward (gradient-capable;
the caller wraps old/ref passes in ``no_grad``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from mm_model import GPT
    from mm_wordle import ConstraintTokenizer


@torch.no_grad()
def sample_group(
    model: GPT,
    prompt_ids: list[int],
    group_size: int,
    device: torch.device,
    letter_mask: torch.Tensor,
    tokenizer: ConstraintTokenizer,
    temperature: float = 1.0,
) -> tuple[list[str], torch.Tensor]:
    """Sample ``group_size`` 5-letter guesses from one prompt (batched, KV-cached).

    Returns (words, token_ids) where token_ids is ``(group_size, 5)``.
    """
    was_training = model.training
    model.eval()
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).expand(group_size, -1)
    prompt_len = prompt.shape[1]

    logits, _, kv = model(prompt)  # prime cache with the full prompt (causal pass)
    cols: list[torch.Tensor] = []
    for i in range(5):
        next_logits = logits[:, -1, :] / temperature + letter_mask
        nxt = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)  # (G, 1)
        cols.append(nxt)
        if i < 4:
            logits, _, kv = model(nxt, kv_cache=kv, start_pos=prompt_len + i)

    gen = torch.cat(cols, dim=1)  # (G, 5)
    words = [tokenizer.decode_letters(row) for row in gen.tolist()]
    if was_training:
        model.train()
    return words, gen


def group_log_probs(
    model: GPT,
    prompt_ids: list[int],
    group_token_ids: torch.Tensor,
    device: torch.device,
    letter_mask: torch.Tensor,
) -> torch.Tensor:
    """Sequence-level (summed) log prob of each group guess given the prompt.

    One batched forward over ``(group_size, prompt_len + 5)``. Gradient flows when
    called outside ``no_grad`` (use that for the PPO current-policy pass; wrap the
    old/ref passes in ``no_grad``). Returns ``(group_size,)``.
    """
    group_size = group_token_ids.shape[0]
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).expand(group_size, -1)
    prompt_len = prompt.shape[1]
    full = torch.cat([prompt, group_token_ids], dim=1)  # (G, P+5)
    logits, _, _ = model(full)
    gen_logits = logits[:, prompt_len - 1 : prompt_len + 4, :] + letter_mask  # (G, 5, V)
    logp = F.log_softmax(gen_logits, dim=-1)
    tok_logp = logp.gather(2, group_token_ids.unsqueeze(-1)).squeeze(-1)  # (G, 5)
    return tok_logp.sum(dim=1)  # (G,)
