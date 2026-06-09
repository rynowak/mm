"""Precomputed feedback-pattern matrix for fast Wordle info-gain and filtering.

Wordle feedback depends only on (guess, target). We precompute, for a guess set
``G`` and a target/answer universe ``U``, an integer matrix ``P[g, t]`` holding the
feedback *pattern id* of guessing ``G[g]`` against answer ``U[t]``. A pattern id
packs the 5 per-letter outcomes (gray=0, yellow=1, green=2) in base 3, so it fits
in a single ``uint8`` (0..242).

With this matrix the three RL hot paths become vectorized numpy:

* expected info gain of a guess over a candidate set = entropy of the pattern
  histogram (one ``np.bincount``);
* candidate filtering after an observed pattern = a boolean column mask;
* best-available-word (the reward normalization denominator) = max info gain over
  a (possibly bounded) set of guess rows.

The vectorized feedback in :func:`_guess_patterns` is an exact, count-based
re-implementation of :meth:`mm_wordle.game.WordleEnv.compute_feedback` (greens
first, then yellows consuming remaining non-green target letters left-to-right).
``tests/test_pattern.py`` asserts bit-for-bit equivalence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Per-letter feedback codes, packed in base 3 (position 0 is least significant).
_GRAY = 0
_YELLOW = 1
_GREEN = 2
N_PATTERNS = 243  # 3 ** 5
_POWERS = np.array([3**i for i in range(5)], dtype=np.int64)

# Pattern id when all 5 letters are green (the solved state): 2*(1+3+9+27+81).
SOLVED_PATTERN = int((_GREEN * _POWERS).sum())


def _words_to_codes(words: list[str]) -> np.ndarray:
    """Encode words as an ``(n, 5)`` int8 array of letter codes (a=0..z=25)."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (arr.reshape(len(words), 5) - ord("a")).astype(np.int8)


def _guess_patterns(g_codes: np.ndarray, t_codes: np.ndarray) -> np.ndarray:
    """Feedback pattern ids of one guess against every target.

    ``g_codes``: ``(5,)`` letter codes for the guess.
    ``t_codes``: ``(n, 5)`` letter codes for the targets.
    Returns ``(n,)`` uint8 pattern ids, matching ``WordleEnv.compute_feedback``.
    """
    n = t_codes.shape[0]
    code = np.zeros((n, 5), dtype=np.int8)

    green = t_codes == g_codes  # (n, 5)
    code[green] = _GREEN

    # avail[row, letter] = count of NON-green target positions holding that letter.
    # (Green positions consume their own letter, so they cancel out of availability.)
    avail = np.zeros((n, 26), dtype=np.int16)
    for pos in range(5):
        not_green = ~green[:, pos]
        rows = np.nonzero(not_green)[0]
        np.add.at(avail, (rows, t_codes[rows, pos]), 1)

    # Assign yellows left-to-right over guess positions, consuming availability.
    for pos in range(5):
        gl = int(g_codes[pos])
        can_yellow = (~green[:, pos]) & (avail[:, gl] > 0)
        code[can_yellow, pos] = _YELLOW
        avail[can_yellow, gl] -= 1

    return (code.astype(np.int64) @ _POWERS).astype(np.uint8)


def encode_pattern(codes: list[int]) -> int:
    """Pack a list of 5 per-letter codes (gray=0, yellow=1, green=2) into an id."""
    return int(sum(c * 3**i for i, c in enumerate(codes)))


def decode_pattern(pattern_id: int) -> list[int]:
    """Unpack a pattern id into a list of 5 per-letter codes."""
    return [(pattern_id // 3**i) % 3 for i in range(5)]


class PatternMatrix:
    """A precomputed ``guesses x targets`` feedback-pattern matrix."""

    def __init__(self, guesses: list[str], targets: list[str], matrix: np.ndarray) -> None:
        if matrix.shape != (len(guesses), len(targets)):
            raise ValueError(f"matrix shape {matrix.shape} != ({len(guesses)}, {len(targets)})")
        self.guesses = guesses
        self.targets = targets
        self.matrix = matrix
        self.guess_index = {w: i for i, w in enumerate(guesses)}
        self.target_index = {w: i for i, w in enumerate(targets)}

    # --- construction / persistence ---

    @classmethod
    def build(cls, guesses: list[str], targets: list[str]) -> PatternMatrix:
        """Build the matrix in memory (vectorized per guess row)."""
        g_codes = _words_to_codes(guesses)
        t_codes = _words_to_codes(targets)
        matrix = np.empty((len(guesses), len(targets)), dtype=np.uint8)
        for gi in range(len(guesses)):
            matrix[gi] = _guess_patterns(g_codes[gi], t_codes)
        return cls(guesses, targets, matrix)

    @classmethod
    def from_words(cls, words: list[str]) -> PatternMatrix:
        """Square matrix where the guess set equals the target universe (V3: G == U)."""
        return cls.build(words, words)

    @classmethod
    def load_or_build(cls, words: list[str], cache_dir: str | Path, mmap: bool = True) -> PatternMatrix:
        """Load a cached square matrix for ``words`` or build and cache it (N1).

        The cache key is a hash of the word list, so a different word set builds a
        fresh artifact. The full 14,855-word matrix is ~220 MB and ~30s to build;
        subsequent runs memory-map it instantly.
        """
        key = hashlib.sha256("\n".join(words).encode("utf-8")).hexdigest()[:16]
        base = Path(cache_dir) / f"pattern_{key}"
        if base.with_suffix(".npy").exists() and base.with_suffix(".meta.json").exists():
            return cls.load(base, mmap=mmap)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        matrix = cls.from_words(words)
        matrix.save(base)
        return matrix

    def save(self, path: str | Path) -> None:
        """Persist the matrix (``.npy``) plus a sidecar listing the word sets."""
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.matrix)
        path.with_suffix(".meta.json").write_text(json.dumps({"guesses": self.guesses, "targets": self.targets}))

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> PatternMatrix:
        """Load a persisted matrix, memory-mapped by default (220 MB at full scale)."""
        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        matrix = np.load(path.with_suffix(".npy"), mmap_mode="r" if mmap else None)
        return cls(meta["guesses"], meta["targets"], matrix)

    # --- queries (index-based; ``*_idx`` take numpy index arrays into ``targets``) ---

    def pattern_id(self, guess: str, target: str) -> int:
        """Feedback pattern id of ``guess`` against ``target``."""
        return int(self.matrix[self.guess_index[guess], self.target_index[target]])

    def expected_info_gain(self, guess: str, candidate_idx: np.ndarray) -> float:
        """Expected info gain (bits) of ``guess`` over the candidate target indices.

        Equals the entropy of the feedback-pattern distribution over the
        candidates; matches ``reward._compute_expected_info_gain``.
        """
        n = candidate_idx.shape[0]
        if n <= 1:
            return 0.0
        row = self.matrix[self.guess_index[guess], candidate_idx]
        counts = np.bincount(row, minlength=N_PATTERNS)
        nz = counts[counts > 0].astype(np.float64)
        p = nz / n
        return float(-(p * np.log2(p)).sum())

    def consistent_idx(self, guess: str, observed_pattern: int, candidate_idx: np.ndarray) -> np.ndarray:
        """Subset of candidate indices whose feedback for ``guess`` equals the observed pattern.

        Equivalent to ``solver.filter_candidates`` (a candidate survives iff it
        would have produced the observed feedback).
        """
        row = self.matrix[self.guess_index[guess], candidate_idx]
        return candidate_idx[row == observed_pattern]

    def best_expected_info_gain(self, candidate_idx: np.ndarray, search_idx: np.ndarray | None = None) -> float:
        """Max expected info gain over the search guesses (default: all guesses).

        For the bounded-search optimization (§5.7-A), pass ``search_idx`` =
        candidates ∪ top-K probes as row indices into ``guesses``.
        """
        n = candidate_idx.shape[0]
        if n <= 1:
            return 0.0
        rows = range(len(self.guesses)) if search_idx is None else search_idx.tolist()
        log2n = np.log2(n)
        best = 0.0
        for gi in rows:
            counts = np.bincount(self.matrix[gi, candidate_idx], minlength=N_PATTERNS)
            nz = counts[counts > 0].astype(np.float64)
            # entropy = log2(n) - (1/n) * sum(c * log2(c))
            ig = log2n - float((nz * np.log2(nz)).sum()) / n
            if ig > best:
                best = ig
        return best
