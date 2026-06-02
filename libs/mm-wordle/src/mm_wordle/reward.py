"""Reward function for Wordle RL training.

Reward = actual_info_gain - expected_info_gain

This normalizes by the expected value, so the reward reflects decision
quality rather than outcome luck. A guess that's good at splitting
candidates scores positive even with unlucky feedback. A lazy guess
scores negative even with lucky feedback.
"""

from __future__ import annotations

import math
from collections import Counter

from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.solver import filter_candidates


def _feedback_pattern(guess: str, target: str) -> tuple[str, ...]:
    """Compute feedback as a hashable tuple for grouping."""
    fb = WordleEnv.compute_feedback(guess, target)
    return tuple(f.value for f in fb)


def expected_info_gain(guess: str, candidates: list[str]) -> float:
    """Compute expected information gain for a guess across all candidates.

    Groups candidates by feedback pattern, then computes the weighted
    average info gain. This is O(N) in the number of candidates —
    no filtering needed.
    """
    n = len(candidates)
    if n <= 1:
        return 0.0

    pattern_counts: Counter[tuple[str, ...]] = Counter()
    for target in candidates:
        pattern = _feedback_pattern(guess, target)
        pattern_counts[pattern] += 1

    expected = 0.0
    for count in pattern_counts.values():
        info = math.log2(n / count)
        expected += (count / n) * info

    return expected


def compute_reward(
    guess: str,
    feedback: list[LetterFeedback],
    candidates_before: list[str],
) -> tuple[float, float, float]:
    """Compute reward as actual info gain minus expected info gain.

    Returns (reward, actual_info_gain, expected_info_gain).
    """
    n_before = len(candidates_before)
    if n_before <= 1:
        return 0.0, 0.0, 0.0

    candidates_after = filter_candidates(candidates_before, guess, feedback)
    n_after = max(len(candidates_after), 1)
    actual = math.log2(n_before / n_after)

    expected = expected_info_gain(guess, candidates_before)

    return actual - expected, actual, expected
