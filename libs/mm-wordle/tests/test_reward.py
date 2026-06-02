"""Tests for reward function."""

import math

from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.reward import compute_reward, expected_info_gain
from mm_wordle.words import load_answers


class TestComputeReward:
    def test_solved_gives_max_info(self) -> None:
        env = WordleEnv()
        answers = load_answers()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "crane")
        fb = state.guesses[-1].feedback
        reward, actual, expected = compute_reward("crane", fb, list(answers))
        assert actual == math.log2(len(answers))
        assert reward > 0

    def test_good_guess_positive_reward(self) -> None:
        env = WordleEnv()
        answers = load_answers()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "slate")
        fb = state.guesses[-1].feedback
        reward, actual, expected = compute_reward("slate", fb, list(answers))
        assert actual > 4.0
        assert expected > 4.0

    def test_bad_guess_negative_reward(self) -> None:
        env = WordleEnv()
        answers = load_answers()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "fuzzy")
        fb = state.guesses[-1].feedback
        reward, actual, expected = compute_reward("fuzzy", fb, list(answers))
        assert actual < 2.0
        assert reward < 0

    def test_returns_three_values(self) -> None:
        env = WordleEnv()
        answers = load_answers()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "slate")
        fb = state.guesses[-1].feedback
        result = compute_reward("slate", fb, list(answers))
        assert len(result) == 3
        reward, actual, expected = result
        assert isinstance(reward, float)
        assert isinstance(actual, float)
        assert isinstance(expected, float)

    def test_single_candidate_returns_zero(self) -> None:
        reward, actual, expected = compute_reward("crane", [LetterFeedback.GREEN] * 5, ["crane"])
        assert reward == 0.0
        assert actual == 0.0
        assert expected == 0.0

    def test_reward_equals_actual_minus_expected(self) -> None:
        env = WordleEnv()
        answers = load_answers()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "arose")
        fb = state.guesses[-1].feedback
        reward, actual, expected = compute_reward("arose", fb, list(answers))
        assert abs(reward - (actual - expected)) < 1e-10


class TestExpectedInfoGain:
    def test_positive(self) -> None:
        answers = load_answers()
        eg = expected_info_gain("slate", list(answers))
        assert eg > 0

    def test_good_guess_higher_than_bad(self) -> None:
        answers = load_answers()
        eg_slate = expected_info_gain("slate", list(answers))
        eg_fuzzy = expected_info_gain("fuzzy", list(answers))
        assert eg_slate > eg_fuzzy

    def test_single_candidate(self) -> None:
        eg = expected_info_gain("crane", ["crane"])
        assert eg == 0.0
