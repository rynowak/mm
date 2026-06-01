"""Group Relative Policy Optimization (GRPO) algorithm."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_group_advantages(rewards: Tensor) -> Tensor:
    """Normalize rewards within a group: (r - mean) / (std + eps).

    Args:
        rewards: (group_size,) rewards for each completion in the group.

    Returns:
        (group_size,) normalized advantages with approximately zero mean
        and unit variance. If all rewards are identical, returns zeros.
    """
    mean = rewards.mean()
    std = rewards.std(correction=0)

    # If std is zero (all same rewards or single element), return zeros
    eps = 1e-8
    if std < eps:
        return torch.zeros_like(rewards)

    advantages = (rewards - mean) / (std + eps)
    return advantages


def grpo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    ref_log_probs: Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
) -> tuple[Tensor, dict[str, float]]:
    """Compute GRPO loss for a single prompt's group of completions.

    The algorithm:
        1. Normalize rewards within group to get advantages.
        2. Compute per-token probability ratios between current and old policy.
        3. Sum log-ratios over sequence to get per-completion ratios.
        4. Apply PPO-style clipping to the surrogate objective.
        5. Add KL divergence penalty from reference policy.

    Args:
        log_probs: (group_size, seq_len) log probs under current policy.
        old_log_probs: (group_size, seq_len) log probs under old policy.
        rewards: (group_size,) rewards for each completion.
        ref_log_probs: (group_size, seq_len) log probs under reference policy.
        clip_epsilon: PPO clipping range.
        beta: KL penalty coefficient.

    Returns:
        (loss, metrics) where metrics contains:
            - policy_loss: the clipped surrogate objective (before KL)
            - kl_div: mean KL divergence from reference policy
            - entropy: mean entropy of current policy
            - advantages_mean: mean of computed advantages
            - advantages_std: std of computed advantages
            - clip_fraction: fraction of ratios that were clipped
    """
    # Step 1: Normalize rewards to get advantages
    advantages = compute_group_advantages(rewards)  # (group_size,)

    # Step 2: Compute per-token log ratios
    log_ratios = log_probs - old_log_probs  # (group_size, seq_len)

    # Step 3: Sum log ratios over sequence to get per-completion log ratio
    # then exponentiate to get the probability ratio
    completion_log_ratios = log_ratios.sum(dim=-1)  # (group_size,)
    ratios = torch.exp(completion_log_ratios)  # (group_size,)

    # Step 4: Clipped surrogate objective
    # unclipped: ratio * advantage
    # clipped: clip(ratio, 1-eps, 1+eps) * advantage
    # take the min (pessimistic bound)
    unclipped = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratios * advantages
    surrogate = torch.min(unclipped, clipped)
    policy_loss = -surrogate.mean()

    # Step 5: KL divergence from reference policy
    # KL(current || ref) = sum over tokens of (current_log_prob - ref_log_prob)
    # averaged over completions
    kl_per_token = log_probs - ref_log_probs  # (group_size, seq_len)
    kl_per_completion = kl_per_token.sum(dim=-1)  # (group_size,)
    kl_div = kl_per_completion.mean()

    # Total loss = policy loss + beta * KL penalty
    loss = policy_loss + beta * kl_div

    # Compute entropy: -sum(p * log_p) approximated as -mean(log_probs)
    entropy = -log_probs.mean()

    # Compute clip fraction: how often was the ratio clipped?
    clip_fraction = ((ratios > 1.0 + clip_epsilon) | (ratios < 1.0 - clip_epsilon)).float().mean().item()

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl_div": kl_div.item(),
        "entropy": entropy.item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
        "clip_fraction": clip_fraction,
    }

    return loss, metrics
