"""Tests for the V3 pattern-matrix RL reward."""

import numpy as np
from mm_wordle import PatternMatrix
from mm_wordle.pattern import SOLVED_PATTERN
from mm_wordle.reward import ENDGAME_BONUS, INFO_GAIN_SCALE, INVALID_WORD_PENALTY, SOLVED_BONUS
from mm_wordle.words import load_full_word_set

from wordle3.reward import compute_reward_v3


def _pm(n: int = 200) -> tuple[PatternMatrix, list[str]]:
    words = load_full_word_set()[:n]
    return PatternMatrix.from_words(words), words


def test_invalid_word_gets_penalty():
    pm, _ = _pm()
    r, ig = compute_reward_v3(pm, "zzzzz", np.arange(len(pm.targets)), 0, composite=True)
    assert r == INVALID_WORD_PENALTY
    assert ig == 0.0


def test_non_composite_is_raw_info_gain():
    pm, words = _pm()
    cand = np.arange(len(words))
    r, ig = compute_reward_v3(pm, words[10], cand, 0, composite=False)
    assert r == ig == pm.expected_info_gain(words[10], cand)


def test_composite_normalized_is_bounded():
    pm, words = _pm()
    cand = np.sort(np.random.default_rng(0).choice(len(words), size=30, replace=False))
    search = np.arange(len(words))
    r, _ = compute_reward_v3(pm, words[3], cand, 0, composite=True, search_idx=search)
    # No bonuses (>2 candidates, not solved): normalized * scale, in [0, scale].
    assert 0.0 <= r <= INFO_GAIN_SCALE + 1e-6


def test_endgame_bonus_when_guess_is_candidate_and_few_left():
    pm, words = _pm()
    gi = pm.guess_index[words[7]]
    cand = np.array([gi, pm.guess_index[words[8]]])  # 2 candidates incl. the guess
    observed = pm.pattern_id(words[7], words[8])  # not solved
    r, _ = compute_reward_v3(pm, words[7], cand, observed, composite=True, search_idx=cand)
    assert r >= ENDGAME_BONUS  # endgame bonus applied (no solve)


def test_solved_bonus_when_correct_and_few_left():
    pm, words = _pm()
    gi = pm.guess_index[words[7]]
    cand = np.array([gi, pm.guess_index[words[8]]])
    r, _ = compute_reward_v3(pm, words[7], cand, SOLVED_PATTERN, composite=True, search_idx=cand)
    assert r >= ENDGAME_BONUS + SOLVED_BONUS  # both bonuses (guess is candidate + solved)
