"""Strong ("golden") Wordle solver backed by the pattern matrix (ADR-8).

``solver.play_game_good`` degenerates to random openers once the candidate set
exceeds 500 words (``solver.py:99``) and ``entropy_guess`` only samples 200
candidates — useless over the 14,855-word universe. This solver instead:

* plays a single precomputed best opener over the full universe U;
* for later turns picks the exact max-info-gain guess over
  ``candidates ∪ top-K global probes`` (no sampling cap);
* resets an emptied candidate set to U (not a restricted answer list).

It is the source of SFT demonstration games and a reference baseline for RL eval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mm_wordle.game import GameState, WordleEnv
    from mm_wordle.pattern import PatternMatrix


class GoldenSolver:
    """Exact info-gain solver over a precomputed pattern matrix."""

    def __init__(self, pattern_matrix: PatternMatrix, probe_top_k: int = 300) -> None:
        self.pm = pattern_matrix
        self.n = len(pattern_matrix.targets)
        self.full_idx = np.arange(self.n)
        # One-time: expected info gain of every guess against the full universe.
        igs = np.array(
            [pattern_matrix.expected_info_gain(g, self.full_idx) for g in pattern_matrix.guesses],
            dtype=np.float64,
        )
        self.best_opener_idx = int(igs.argmax())
        self.top_probes = np.argsort(igs)[-probe_top_k:].astype(np.int64)

    @property
    def best_opener(self) -> str:
        return self.pm.guesses[self.best_opener_idx]

    def choose_guess(self, candidate_idx: np.ndarray) -> str:
        """Pick the next guess for the given remaining-candidate indices.

        Full-lexicon one-step max-info-gain search (vectorized), which solves ~100%
        of this answer set — vs the old bounded `candidates ∪ top-300` search, which
        missed mid-game separating probes and only reached ~96%.
        """
        n = len(candidate_idx)
        if n >= self.n:
            return self.best_opener
        if n <= 2:
            return self.pm.targets[int(candidate_idx[0])]
        return self.pm.guesses[self.pm.best_guess_idx(candidate_idx)]


def play_golden_game(solver: GoldenSolver, env: WordleEnv, target: str) -> GameState:
    """Play one game to completion with the golden solver; return the final state."""
    state = env.reset(target_word=target)
    candidate_idx = solver.full_idx.copy()
    while not state.solved and not state.failed:
        guess = solver.choose_guess(candidate_idx)
        state, _ = env.step(state, guess)
        observed = solver.pm.pattern_id(guess, target)
        candidate_idx = solver.pm.consistent_idx(guess, observed, candidate_idx)
        if len(candidate_idx) == 0:  # inconsistency fallback -> reset to U (ADR-8)
            candidate_idx = solver.full_idx.copy()
    return state
