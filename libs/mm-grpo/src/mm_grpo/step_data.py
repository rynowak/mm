"""Step data assembly for GRPO training visualization."""

from __future__ import annotations

from mm_viz import CompletionData, GRPOStepData


def build_step_data(
    step: int,
    game_state_tokens: list[str],
    game_state_text: str,
    completion_texts: list[str],
    completion_token_lists: list[list[str]],
    log_probs_per_completion: list[list[float]],
    rewards: list[float],
    reward_breakdowns: list[dict[str, float]],
    advantages: list[float],
    group_mean: float,
    group_std: float,
    old_probs: list[float],
    new_probs: list[float],
    kl_divergence: float,
) -> GRPOStepData:
    """Assemble a GRPOStepData for visualization.

    Takes the raw outputs from a GRPO training step and packages them into
    the structured GRPOStepData format used by mm-viz.

    Args:
        step: Training step number.
        game_state_tokens: Tokenized game state (prompt).
        game_state_text: Human-readable game state text.
        completion_texts: Text of each completion in the group.
        completion_token_lists: Tokens for each completion.
        log_probs_per_completion: Per-token log probs for each completion.
        rewards: Scalar reward for each completion.
        reward_breakdowns: Per-component reward breakdown for each completion.
        advantages: Normalized advantage for each completion.
        group_mean: Mean reward across the group.
        group_std: Std of rewards across the group.
        old_probs: Probability of each completion under old policy.
        new_probs: Probability of each completion under new policy.
        kl_divergence: KL divergence between new and reference policy.

    Returns:
        A fully populated GRPOStepData instance.
    """
    completions = [
        CompletionData(
            tokens=tokens,
            text=text,
            log_probs=lp,
            reward=reward,
            reward_breakdown=breakdown,
        )
        for tokens, text, lp, reward, breakdown in zip(
            completion_token_lists,
            completion_texts,
            log_probs_per_completion,
            rewards,
            reward_breakdowns,
            strict=True,
        )
    ]

    return GRPOStepData(
        step=step,
        game_state_tokens=game_state_tokens,
        game_state_text=game_state_text,
        completions=completions,
        rewards=rewards,
        advantages=advantages,
        group_mean=group_mean,
        group_std=group_std,
        old_probs=old_probs,
        new_probs=new_probs,
        kl_divergence=kl_divergence,
    )
