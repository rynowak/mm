"""Tests for reward function."""

from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.reward import (
    ENDGAME_BONUS,
    INFO_GAIN_SCALE,
    INVALID_WORD_PENALTY,
    SOLVED_BONUS,
    compute_reward,
    expected_info_gain,
)
from mm_wordle.words import load_answers


class TestNonCompositeReward:
    def test_reward_equals_expected_info_gain(self) -> None:
        answers = load_answers()
        fb = WordleEnv.compute_feedback("arose", "crane")
        reward, _, expected = compute_reward("arose", fb, list(answers))
        assert abs(reward - expected) < 1e-10

    def test_good_guess_higher_than_bad(self) -> None:
        answers = load_answers()
        fb_good = WordleEnv.compute_feedback("slate", "crane")
        fb_bad = WordleEnv.compute_feedback("fuzzy", "crane")
        r_good, _, _ = compute_reward("slate", fb_good, list(answers))
        r_bad, _, _ = compute_reward("fuzzy", fb_bad, list(answers))
        assert r_good > r_bad


class TestCompositeReward:
    def test_normalized_ig_scales_to_info_gain_scale(self) -> None:
        """A perfect info gain guess should score close to INFO_GAIN_SCALE."""
        candidates = ["crane", "crate"]
        fb = WordleEnv.compute_feedback("crane", "crane")
        reward, _, _ = compute_reward("crane", fb, candidates, composite=True)
        # With 2 candidates, max_possible = 1 bit. A perfect split gets normalized=1.0.
        # Plus endgame bonus + solve bonus since candidates=2 and solved.
        assert reward >= INFO_GAIN_SCALE * 0.5

    def test_opener_info_gain_dominates(self) -> None:
        """At turn 1, info gain should be the dominant reward."""
        answers = load_answers()
        fb = WordleEnv.compute_feedback("slate", "crane")
        reward, _, _ = compute_reward("slate", fb, list(answers), composite=True)
        # slate gets ~5.86/11.2 = 0.52 normalized, * 10 = 5.2
        assert reward > 4.0
        assert reward < INFO_GAIN_SCALE

    def test_midgame_no_solve_bonus(self) -> None:
        """Solving with 3+ candidates should NOT get solve bonus."""
        candidates = ["crane", "crate", "craze"]
        fb = [LetterFeedback.GREEN] * 5
        reward, _, _ = compute_reward("crane", fb, candidates, composite=True)
        # No solve bonus (n_before=3 > 2), just normalized info gain
        assert reward < INFO_GAIN_SCALE

    def test_midgame_discovery_competes(self) -> None:
        """Discovery words should compete on normalized info gain."""
        candidates = ["crane", "crate", "craze"]
        fb_disc = WordleEnv.compute_feedback("plait", "crane")
        r_disc, _, _ = compute_reward("plait", fb_disc, candidates, composite=True)
        # Discovery word should get positive reward from info gain
        assert r_disc > 0

    def test_endgame_bonus_at_2_candidates(self) -> None:
        candidates = ["crane", "crate"]
        fb = WordleEnv.compute_feedback("crane", "crate")
        reward, _, _ = compute_reward("crane", fb, candidates, composite=True)
        assert reward >= ENDGAME_BONUS

    def test_endgame_bonus_at_1_candidate(self) -> None:
        candidates = ["crane"]
        fb = [LetterFeedback.GREEN] * 5
        reward, _, _ = compute_reward("crane", fb, candidates, composite=True)
        assert reward >= ENDGAME_BONUS + SOLVED_BONUS

    def test_no_endgame_bonus_at_3_candidates(self) -> None:
        candidates = ["crane", "crate", "craze"]
        fb = WordleEnv.compute_feedback("crane", "crate")
        reward, _, _ = compute_reward("crane", fb, candidates, composite=True)
        ig = expected_info_gain("crane", candidates)
        import math

        normalized = ig / math.log2(len(candidates)) * INFO_GAIN_SCALE
        # Should be just normalized info gain, no bonus
        assert abs(reward - normalized) < 1e-6

    def test_solve_bonus_only_endgame(self) -> None:
        """Solve bonus only fires when candidates <= 2."""
        # Solve with many candidates — no bonus
        answers = load_answers()
        fb_solved = [LetterFeedback.GREEN] * 5
        r_lucky, _, _ = compute_reward("crane", fb_solved, list(answers), composite=True)

        # Solve with 1 candidate — bonus
        r_skill, _, _ = compute_reward("crane", fb_solved, ["crane"], composite=True)

        assert r_skill > r_lucky

    def test_grpo_variance_endgame(self) -> None:
        candidates = ["crane"]
        fb_right = WordleEnv.compute_feedback("crane", "crane")
        fb_wrong = WordleEnv.compute_feedback("slate", "crane")
        r_right, _, _ = compute_reward("crane", fb_right, candidates, composite=True)
        r_wrong, _, _ = compute_reward("slate", fb_wrong, candidates, composite=True)
        assert r_right > r_wrong

    def test_invalid_word_gets_penalty_composite(self) -> None:
        answers = load_answers()
        fb = [LetterFeedback.GRAY] * 5
        reward, _, _ = compute_reward("zzzzz", fb, list(answers), composite=True)
        assert reward == INVALID_WORD_PENALTY

    def test_invalid_word_gets_penalty_non_composite(self) -> None:
        answers = load_answers()
        fb = [LetterFeedback.GRAY] * 5
        reward, _, _ = compute_reward("folka", fb, list(answers), composite=False)
        assert reward == INVALID_WORD_PENALTY

    def test_invalid_word_worse_than_any_valid(self) -> None:
        answers = load_answers()
        fb_invalid = [LetterFeedback.GRAY] * 5
        fb_valid = WordleEnv.compute_feedback("fuzzy", "crane")
        r_invalid, _, _ = compute_reward("folka", fb_invalid, list(answers), composite=True)
        r_valid, _, _ = compute_reward("fuzzy", fb_valid, list(answers), composite=True)
        assert r_valid > r_invalid


class TestExpectedInfoGain:
    def test_positive(self) -> None:
        answers = load_answers()
        assert expected_info_gain("slate", list(answers)) > 0

    def test_good_higher_than_bad(self) -> None:
        answers = load_answers()
        assert expected_info_gain("slate", list(answers)) > expected_info_gain("fuzzy", list(answers))

    def test_single_candidate_zero(self) -> None:
        assert expected_info_gain("crane", ["crane"]) == 0.0
