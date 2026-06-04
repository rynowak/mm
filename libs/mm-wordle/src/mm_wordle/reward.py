"""Reward function for Wordle RL training.

See docs/reward-function.md for the full design.
"""

from __future__ import annotations

import math
from collections import Counter

from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.solver import filter_candidates
from mm_wordle.words import all_valid_words, load_answers

_ANSWERS = load_answers()
_VALID_WORDS = all_valid_words()

INFO_GAIN_SCALE = 10.0
ENDGAME_BONUS = 3.0
SOLVED_BONUS = 5.0
INVALID_WORD_PENALTY = -10.0


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
    """Precomputed expected info gain for the full answer list."""

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
        answers = list(_ANSWERS)
        for word in answers:
            self._full_list_cache[word] = _compute_expected_info_gain(word, answers)


_CACHE = ExpectedInfoGainCache()


def precompute_expected_info_gain() -> None:
    _CACHE.precompute()


def expected_info_gain(guess: str, candidates: list[str]) -> float:
    return _CACHE.get(guess, candidates)


def compute_reward(
    guess: str,
    feedback: list[LetterFeedback],
    candidates_before: list[str],
    composite: bool = False,
) -> tuple[float, float, float]:
    """Compute reward for a guess.

    Returns (reward, actual_info_gain, expected_info_gain).

    When composite=False: reward is pure expected info gain (unnormalized).
    When composite=True: reward is normalized info gain + endgame bonus + solve bonus.
    """
    n_before = len(candidates_before)
    solved = all(f == LetterFeedback.GREEN for f in feedback)

    if n_before <= 1:
        actual = SOLVED_BONUS if solved else 0.0
    else:
        candidates_after = filter_candidates(candidates_before, guess, feedback)
        n_after = max(len(candidates_after), 1)
        actual = math.log2(n_before / n_after)

    expected = expected_info_gain(guess, candidates_before)

    if not composite:
        return expected, actual, expected

    if guess not in _VALID_WORDS:
        return INVALID_WORD_PENALTY, actual, expected

    if n_before > 1:
        max_possible = math.log2(n_before)
        normalized = expected / max_possible if max_possible > 0 else 0.0
        reward = normalized * INFO_GAIN_SCALE
    else:
        reward = 0.0

    if n_before <= 2 and guess in candidates_before:
        reward += ENDGAME_BONUS
    if solved and n_before <= 2:
        reward += SOLVED_BONUS

    return reward, actual, expected
