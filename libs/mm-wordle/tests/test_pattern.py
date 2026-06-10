"""Equivalence + correctness tests for the precomputed pattern matrix.

The pattern matrix is the V3 reward/solver backend; these tests are the Phase-1
gate: the vectorized path must match ``compute_feedback`` / ``reward`` / ``solver``
exactly on real word sets.
"""

import time

import numpy as np
import pytest
from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.pattern import PatternMatrix, decode_pattern, encode_pattern
from mm_wordle.reward import _compute_expected_info_gain
from mm_wordle.solver import filter_candidates
from mm_wordle.words import load_full_word_set

_CODE = {LetterFeedback.GRAY: 0, LetterFeedback.YELLOW: 1, LetterFeedback.GREEN: 2}


def _ref_pattern_id(guess: str, target: str) -> int:
    return encode_pattern([_CODE[f] for f in WordleEnv.compute_feedback(guess, target)])


# Words chosen to stress duplicate-letter handling (greens-first, then yellows).
TRICKY = ["foggy", "ovoid", "slate", "humor", "array", "tarot", "sissy", "esses", "abyss", "llama", "geese", "added"]


@pytest.fixture(scope="module")
def small_matrix():
    words = load_full_word_set()[:600]
    return PatternMatrix.from_words(words)


def test_encode_decode_roundtrip():
    for pid in range(243):
        assert encode_pattern(decode_pattern(pid)) == pid


def test_feedback_matches_compute_feedback_tricky():
    m = PatternMatrix.from_words(TRICKY)
    for g in TRICKY:
        for t in TRICKY:
            assert m.pattern_id(g, t) == _ref_pattern_id(g, t), f"mismatch {g} vs {t}"


def test_feedback_matches_compute_feedback_bulk():
    """Every cell of a 200x200 matrix decodes to the reference feedback."""
    words = load_full_word_set()[:200]
    m = PatternMatrix.from_words(words)
    for gi, g in enumerate(words):
        for ti, t in enumerate(words):
            assert int(m.matrix[gi, ti]) == _ref_pattern_id(g, t)


def test_expected_info_gain_matches_reward(small_matrix):
    words = small_matrix.targets
    rng = np.random.default_rng(0)
    for _ in range(40):
        guess = words[int(rng.integers(len(words)))]
        k = int(rng.integers(2, len(words)))
        cand_idx = np.sort(rng.choice(len(words), size=k, replace=False))
        cand_words = [words[i] for i in cand_idx]
        got = small_matrix.expected_info_gain(guess, cand_idx)
        want = _compute_expected_info_gain(guess, cand_words)
        assert got == pytest.approx(want, abs=1e-9), f"{guess}: {got} != {want}"


def test_expected_info_gain_single_candidate_is_zero(small_matrix):
    words = small_matrix.targets
    assert small_matrix.expected_info_gain(words[3], np.array([7])) == 0.0


def test_consistent_idx_matches_ground_truth(small_matrix):
    """A candidate survives iff it would have produced the observed feedback.

    Ground truth is the actual game (``compute_feedback``); ``filter_candidates``
    now matches this too (its duplicate-letter bug was fixed).
    """
    words = small_matrix.targets
    rng = np.random.default_rng(1)
    all_idx = np.arange(len(words))
    for _ in range(40):
        guess = words[int(rng.integers(len(words)))]
        target = words[int(rng.integers(len(words)))]
        feedback = WordleEnv.compute_feedback(guess, target)
        observed = small_matrix.pattern_id(guess, target)

        got = {words[i] for i in small_matrix.consistent_idx(guess, observed, all_idx)}
        want = {w for w in words if WordleEnv.compute_feedback(guess, w) == feedback}
        assert got == want, f"filter mismatch for guess={guess} target={target}"


def test_consistent_idx_duplicate_letter_regression():
    """green+yellow of the same letter implies >=2 of that letter in the answer.

    ``aliya`` vs ``alack`` -> a(green) l(green) i/y(gray) a(yellow): the trailing
    yellow 'a' requires a *second* 'a'. ``allus`` has only one 'a' and must be
    excluded by both the matrix (= the real game) and ``filter_candidates``.
    """
    words = ["aliya", "alack", "allus", "aloft", "abaca"]
    m = PatternMatrix.from_words(words)
    feedback = WordleEnv.compute_feedback("aliya", "alack")
    observed = m.pattern_id("aliya", "alack")
    keep = {words[i] for i in m.consistent_idx("aliya", observed, np.arange(len(words)))}

    assert "alack" in keep
    assert "allus" not in keep  # correct (one 'a')
    assert "allus" not in set(filter_candidates(words, "aliya", feedback))  # fixed reference agrees


def test_best_expected_info_gain_matches_bruteforce():
    words = load_full_word_set()[:150]
    m = PatternMatrix.from_words(words)
    rng = np.random.default_rng(2)
    all_idx = np.arange(len(words))
    for _ in range(10):
        k = int(rng.integers(2, len(words)))
        cand_idx = np.sort(rng.choice(len(words), size=k, replace=False))
        cand_words = [words[i] for i in cand_idx]
        got = m.best_expected_info_gain(cand_idx, search_idx=all_idx)
        want = max(_compute_expected_info_gain(w, cand_words) for w in words)
        assert got == pytest.approx(want, abs=1e-9)


def test_save_load_roundtrip(tmp_path):
    words = load_full_word_set()[:80]
    m = PatternMatrix.from_words(words)
    base = tmp_path / "pm"
    m.save(base)
    loaded = PatternMatrix.load(base)
    assert loaded.guesses == m.guesses
    assert loaded.targets == m.targets
    assert np.array_equal(np.asarray(loaded.matrix), m.matrix)


def test_load_or_build_caches(tmp_path):
    words = load_full_word_set()[:60]
    m1 = PatternMatrix.load_or_build(words, tmp_path)
    assert list(tmp_path.glob("pattern_*.npy"))  # artifact written
    m2 = PatternMatrix.load_or_build(words, tmp_path)  # served from cache
    assert np.array_equal(np.asarray(m1.matrix), np.asarray(m2.matrix))


def test_best_info_gain_faster_than_naive():
    """Pattern matrix best-IG should be dramatically faster than the Python loop."""
    words = load_full_word_set()[:400]
    m = PatternMatrix.from_words(words)
    cand_idx = np.arange(len(words))
    search_idx = np.arange(len(words))

    t0 = time.perf_counter()
    fast = m.best_expected_info_gain(cand_idx, search_idx=search_idx)
    fast_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    slow = max(_compute_expected_info_gain(w, words) for w in words)
    slow_t = time.perf_counter() - t0

    assert fast == pytest.approx(slow, abs=1e-9)
    assert fast_t < slow_t / 2, f"matrix {fast_t:.4f}s not >2x faster than naive {slow_t:.4f}s"


def test_best_guess_idx_matches_bruteforce():
    """The vectorized best-guess picks a guess achieving the true max info gain."""
    words = load_full_word_set()[:150]
    m = PatternMatrix.from_words(words)
    rng = np.random.default_rng(3)
    for _ in range(8):
        k = int(rng.integers(3, len(words)))
        cand = np.sort(rng.choice(len(words), size=k, replace=False))
        gi = m.best_guess_idx(cand)
        chosen_ig = m.expected_info_gain(m.guesses[gi], cand)
        best_ig = max(m.expected_info_gain(w, cand) for w in words)
        assert chosen_ig == pytest.approx(best_ig, abs=1e-9)


def test_best_guess_idx_chunking_is_consistent():
    words = load_full_word_set()[:120]
    m = PatternMatrix.from_words(words)
    cand = np.arange(2, 90)
    # chunked vs single-block must agree on the achieved info gain
    a = m.expected_info_gain(m.guesses[m.best_guess_idx(cand, chunk=16)], cand)
    b = m.expected_info_gain(m.guesses[m.best_guess_idx(cand, chunk=10_000)], cand)
    assert a == pytest.approx(b, abs=1e-9)
