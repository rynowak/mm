"""Reward function for Wordle RL training."""

import math

from mm_wordle.game import LetterFeedback
from mm_wordle.solver import filter_candidates


def compute_reward(
    guess: str,
    feedback: list[LetterFeedback],
    candidates_before: list[str],
) -> float:
    """Compute reward as information gain in bits.

    reward = log2(candidates_before / candidates_after)

    This is the only signal. Solving the puzzle is the ultimate
    information gain (candidates → 1), so it naturally receives
    the highest reward. No special cases needed.
    """
    n_before = len(candidates_before)
    if n_before <= 1:
        return 0.0

    candidates_after = filter_candidates(candidates_before, guess, feedback)
    n_after = max(len(candidates_after), 1)
    return math.log2(n_before / n_after)
