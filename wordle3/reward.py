"""V3 RL reward, pattern-matrix backed (§5.4, mirrors mm_wordle.reward.compute_reward).

Phase 1 (openers): raw expected info gain. Phase 2 (composite): info gain
normalized against the best available guess over G, scaled, plus endgame/solve
bonuses; invalid words get the penalty. Index-based for speed over the 14,855-word
universe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm_wordle.pattern import SOLVED_PATTERN
from mm_wordle.reward import ENDGAME_BONUS, INFO_GAIN_SCALE, INVALID_WORD_PENALTY, SOLVED_BONUS

if TYPE_CHECKING:
    import numpy as np
    from mm_wordle import PatternMatrix


def compute_reward_v3(
    pattern_matrix: PatternMatrix,
    guess: str,
    candidate_idx: np.ndarray,
    observed_pattern: int,
    *,
    composite: bool,
    search_idx: np.ndarray | None = None,
    best_info_gain: float | None = None,
) -> tuple[float, float]:
    """Return (reward, expected_info_gain) for ``guess`` against the candidate set.

    ``observed_pattern`` is the actual feedback pattern id (vs the true target).
    The normalization denominator (best available word) depends only on the
    candidate set, so callers scoring a whole GRPO group should compute it once and
    pass it via ``best_info_gain``; otherwise ``search_idx`` bounds the search
    (candidates ∪ top probes). Both only matter when ``composite``.
    """
    gi = pattern_matrix.guess_index.get(guess)
    if gi is None:
        return INVALID_WORD_PENALTY, 0.0

    n_before = len(candidate_idx)
    expected = pattern_matrix.expected_info_gain(guess, candidate_idx)
    if not composite:
        return expected, expected

    if n_before > 1:
        best = best_info_gain
        if best is None:
            best = pattern_matrix.best_expected_info_gain(candidate_idx, search_idx=search_idx)
        reward = (expected / best) * INFO_GAIN_SCALE if best > 0 else 0.0
    else:
        reward = 0.0

    guess_is_candidate = bool((candidate_idx == gi).any())
    if n_before <= 2 and guess_is_candidate:
        reward += ENDGAME_BONUS
    if observed_pattern == SOLVED_PATTERN and n_before <= 2:
        reward += SOLVED_BONUS
    return reward, expected
