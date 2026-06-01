"""REINFORCE policy gradient with optional baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def reinforce_loss(
    log_probs: Tensor,
    rewards: Tensor,
    baseline: Tensor | None = None,
) -> Tensor:
    """Compute REINFORCE loss: -mean(advantage * sum(log_probs)).

    Args:
        log_probs: (batch, seq_len) log probabilities of chosen actions.
        rewards: (batch,) reward per trajectory.
        baseline: (batch,) or scalar baseline to subtract from rewards.

    Returns:
        Scalar loss tensor.
    """
    # Compute per-trajectory log probability (sum over tokens)
    trajectory_log_probs = log_probs.sum(dim=-1)  # (batch,)

    # Compute advantages: subtract baseline if provided
    advantages = rewards - baseline if baseline is not None else rewards

    # REINFORCE loss: negative because we want to maximize expected reward
    loss = -(advantages * trajectory_log_probs).mean()
    return loss


class MovingAverageBaseline:
    """Exponential moving average baseline for variance reduction.

    Tracks a running average of observed rewards using momentum-based
    smoothing: value = momentum * value + (1 - momentum) * reward.
    """

    def __init__(self, momentum: float = 0.99) -> None:
        self.value: float = 0.0
        self.momentum = momentum
        self._n_updates: int = 0

    def update(self, reward: float) -> None:
        """Update the baseline with a new observed reward."""
        self.value = self.momentum * self.value + (1.0 - self.momentum) * reward
        self._n_updates += 1

    def get(self) -> float:
        """Return bias-corrected baseline value (like Adam)."""
        if self._n_updates == 0:
            return 0.0
        correction = 1.0 - self.momentum**self._n_updates
        return self.value / correction
