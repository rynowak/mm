"""Utility functions for computing log probabilities from model outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sequence_log_probs(logits: Tensor, tokens: Tensor) -> Tensor:
    """Compute per-token log probabilities for a sequence.

    For each position, computes log_softmax over the vocab dimension and
    then gathers the log probability of the actual token that was chosen.

    Args:
        logits: (batch, seq_len, vocab_size) raw model output logits.
        tokens: (batch, seq_len) token IDs that were chosen.

    Returns:
        (batch, seq_len) log probabilities of the chosen tokens.
    """
    # Compute log softmax over vocab dimension
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

    # Gather the log prob of the actual token at each position
    # tokens shape: (batch, seq_len) -> unsqueeze to (batch, seq_len, 1)
    token_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1))

    # Squeeze back to (batch, seq_len)
    return token_log_probs.squeeze(-1)


def collect_completions_log_probs(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
) -> Tensor:
    """Run model on prompt + each completion and extract completion log probs.

    For each completion, concatenates the prompt with the completion tokens,
    runs the model to get logits, and extracts the log probabilities for
    only the completion portion.

    Note: The logits at position i predict the token at position i+1. So to
    get the log prob of completion token j, we use the logits at position
    (prompt_len - 1 + j).

    Args:
        model: A model with forward(idx) -> (logits, loss) interface.
        prompt_ids: (prompt_len,) the prompt/game state token IDs.
        completion_ids: (group_size, completion_len) the generated completions.

    Returns:
        (group_size, completion_len) log probabilities for each completion token.
    """
    group_size, completion_len = completion_ids.shape
    prompt_len = prompt_ids.shape[0]

    # Expand prompt to match group size: (group_size, prompt_len)
    prompt_expanded = prompt_ids.unsqueeze(0).expand(group_size, -1)

    # Concatenate prompt + completion: (group_size, prompt_len + completion_len)
    full_sequences = torch.cat([prompt_expanded, completion_ids], dim=-1)

    # Run model forward pass
    with torch.no_grad():
        logits, _ = model(full_sequences)  # (group_size, total_len, vocab_size)

    # Extract logits that predict the completion tokens.
    # Logits at position (prompt_len - 1) predict the first completion token.
    # Logits at position (prompt_len - 1 + completion_len - 1) predict the last.
    completion_logits = logits[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]

    # Compute log probs for the actual completion tokens
    return sequence_log_probs(completion_logits, completion_ids)
