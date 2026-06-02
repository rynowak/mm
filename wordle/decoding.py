"""Constrained and unconstrained decoding for Wordle guesses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mm_grpo import sequence_log_probs
from torch import Tensor

if TYPE_CHECKING:
    from mm_model import GPT
    from mm_tokenizers import CharTokenizer
    from mm_wordle import WordTrie


def sample_constrained(
    model: GPT,
    game_state_ids: Tensor,
    trie: WordTrie,
    tokenizer: CharTokenizer,
    device: torch.device,
    n_samples: int = 1,
    temperature: float = 1.0,
) -> list[tuple[str, Tensor]]:
    """Sample word(s) using batched trie-constrained autoregressive decoding.

    Uses KV caching and precomputed GPU trie masks for minimal round trips.
    """
    was_training = model.training
    model.eval()

    prompt = game_state_ids.unsqueeze(0).expand(n_samples, -1).to(device)
    prefixes = [""] * n_samples
    all_tokens: list[Tensor] = []

    logits, _, kv_cache = model(prompt)
    logits = logits[:, -1, :] / temperature
    logits = logits + trie.gpu_mask(prefixes)
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    all_tokens.append(next_tokens)

    for pos in range(4):
        token_ids_cpu = all_tokens[-1].cpu().tolist()
        for i in range(n_samples):
            prefixes[i] += tokenizer.decode([token_ids_cpu[i][0]])

        logits, _, kv_cache = model(next_tokens, kv_cache=kv_cache, start_pos=prompt.size(1) + pos)
        logits = logits[:, -1, :] / temperature
        logits = logits + trie.gpu_mask(prefixes)
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        all_tokens.append(next_tokens)

    token_ids_cpu = all_tokens[-1].cpu().tolist()
    for i in range(n_samples):
        prefixes[i] += tokenizer.decode([token_ids_cpu[i][0]])

    word_tokens = torch.cat(all_tokens, dim=1)
    results: list[tuple[str, Tensor]] = []
    for i in range(n_samples):
        word_ids = word_tokens[i]
        word = prefixes[i].ljust(5, "a")[:5]
        results.append((word, word_ids))

    if was_training:
        model.train()

    return results


def sample_unconstrained(
    model: GPT,
    game_state_ids: Tensor,
    device: torch.device,
    tokenizer: CharTokenizer,
    n_samples: int = 1,
    temperature: float = 0.8,
) -> list[tuple[str, Tensor]]:
    """Generate word(s) autoregressively (5 characters each)."""
    was_training = model.training
    model.eval()

    results: list[tuple[str, Tensor]] = []
    prompt = game_state_ids.unsqueeze(0).to(device)

    for _ in range(n_samples):
        output = model.generate(prompt, max_new_tokens=5, temperature=temperature)
        word_ids = output[0, -5:]
        try:
            word = tokenizer.decode(word_ids.tolist())
        except ValueError:
            word = "?????"
        results.append((word, word_ids))

    if was_training:
        model.train()

    return results


def compute_guess_log_probs(
    model: GPT,
    game_state_ids: Tensor,
    word_ids: Tensor,
) -> Tensor:
    """Compute per-token log probs of a guess given a game state.

    Returns (5,) log probabilities for each character of the guess.
    """
    prompt_len = game_state_ids.shape[0]
    full_seq = torch.cat([game_state_ids, word_ids]).unsqueeze(0)
    logits, _, _ = model(full_seq)
    completion_logits = logits[:, prompt_len - 1 : prompt_len - 1 + 5, :]
    lp = sequence_log_probs(completion_logits, word_ids.unsqueeze(0))
    return lp.squeeze(0)
