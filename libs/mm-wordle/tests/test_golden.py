"""Tests for the pattern-matrix golden solver (ADR-8)."""

import numpy as np
from mm_wordle.game import WordleEnv
from mm_wordle.golden import GoldenSolver, play_golden_game
from mm_wordle.pattern import PatternMatrix
from mm_wordle.solver import play_game_good
from mm_wordle.words import load_full_word_set


def _solver(n_words: int = 400) -> tuple[GoldenSolver, PatternMatrix, list[str]]:
    words = load_full_word_set()[:n_words]
    pm = PatternMatrix.from_words(words)
    return GoldenSolver(pm, probe_top_k=50), pm, words


def test_best_opener_is_a_valid_word():
    solver, _, words = _solver()
    assert solver.best_opener in set(words)


def test_choose_guess_on_full_universe_is_best_opener():
    solver, _, _ = _solver()
    assert solver.choose_guess(solver.full_idx) == solver.best_opener


def test_choose_guess_two_candidates_returns_a_candidate():
    solver, pm, words = _solver()
    cand = np.array([3, 17])
    assert solver.choose_guess(cand) in {words[3], words[17]}


def test_golden_games_solve_most_targets():
    solver, _, words = _solver()
    env = WordleEnv()
    targets = words[:60]
    solved = sum(play_golden_game(solver, env, t).solved for t in targets)
    # A strong info-gain solver should win the vast majority within 6 guesses.
    assert solved / len(targets) >= 0.9


def test_golden_beats_play_game_good_on_large_set():
    """Over a >500-word universe, play_game_good plays random openers; golden wins more."""
    words = load_full_word_set()[:600]
    pm = PatternMatrix.from_words(words)
    solver = GoldenSolver(pm, probe_top_k=80)
    env = WordleEnv()
    targets = words[:50]

    golden_solved = sum(play_golden_game(solver, env, t).solved for t in targets)
    good_solved = sum(play_game_good(env, t, words, words).solved for t in targets)
    assert golden_solved >= good_solved


def test_golden_guesses_are_valid_and_consistent():
    solver, pm, words = _solver()
    env = WordleEnv()
    state = play_golden_game(solver, env, words[5])
    valid = set(words)
    for gf in state.guesses:
        assert gf.guess in valid
