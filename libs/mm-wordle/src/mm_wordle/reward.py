"""Reward function for Wordle RL training.

Reward = actual_info_gain - expected_info_gain, with a solve bonus.

See docs/reward-function.md for the full design.
"""

from __future__ import annotations

import math
from collections import Counter

from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.solver import filter_candidates
from mm_wordle.words import load_answers

_ANSWERS = load_answers()
SOLVED_BONUS = math.log2(len(_ANSWERS))


def _feedback_pattern(guess: str, target: str) -> tuple[str, ...]:
    fb = WordleEnv.compute_feedback(guess, target)
    return tuple(f.value for f in fb)


def _compute_expected_info_gain(guess: str, candidates: list[str]) -> float:
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


class ExpectedInfoGainCache:
    """Precomputed expected info gain for the full answer list.

    Turn 1 always uses the same 2,315 candidates, so expected info gain
    for any word is constant. Precompute once at startup (~6s), then
    dict lookup on turn 1 instead of recomputing every time.
    """

    def __init__(self) -> None:
        self._full_list_cache: dict[str, float] = {}
        self._full_list_key = tuple(_ANSWERS)

    def get(self, guess: str, candidates: list[str]) -> float:
        if tuple(candidates) == self._full_list_key:
            if guess not in self._full_list_cache:
                self._full_list_cache[guess] = _compute_expected_info_gain(guess, candidates)
            return self._full_list_cache[guess]
        return _compute_expected_info_gain(guess, candidates)

    def precompute(self) -> None:
        """Precompute expected info gain for all answer words."""
        answers = list(_ANSWERS)
        for word in answers:
            self._full_list_cache[word] = _compute_expected_info_gain(word, answers)


_CACHE = ExpectedInfoGainCache()


def precompute_expected_info_gain() -> None:
    """Call at startup to precompute turn-1 expected info gain for all words."""
    _CACHE.precompute()


def expected_info_gain(guess: str, candidates: list[str]) -> float:
    """Get expected info gain, using cache for the full answer list."""
    return _CACHE.get(guess, candidates)


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
        if all(f == LetterFeedback.GREEN for f in feedback):
            return SOLVED_BONUS, SOLVED_BONUS, 0.0
        return 0.0, 0.0, 0.0

    candidates_after = filter_candidates(candidates_before, guess, feedback)
    n_after = max(len(candidates_after), 1)
    actual = math.log2(n_before / n_after)

    if all(f == LetterFeedback.GREEN for f in feedback):
        actual = max(actual, SOLVED_BONUS)

    expected = expected_info_gain(guess, candidates_before)

    return actual - expected, actual, expected
