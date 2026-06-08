"""Tests for Wordle solvers and transcript generation."""

from mm_tokenizers import CharTokenizer
from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.solver import filter_candidates, play_game_decent, play_game_good, play_game_random
from mm_wordle.transcripts import generate_examples
from mm_wordle.words import load_answers


class TestFilterCandidates:
    def test_green_filters_position(self) -> None:
        candidates = ["crane", "brave", "crime"]
        # Guess "crimp": c is green at position 0, r is green at position 1
        fb = [LetterFeedback.GREEN, LetterFeedback.GREEN, LetterFeedback.GRAY, LetterFeedback.GRAY, LetterFeedback.GRAY]
        result = filter_candidates(candidates, "crimp", fb)
        assert "crane" in result  # c at 0, r at 1 — matches
        assert "brave" not in result  # b at 0 — doesn't match green c

    def test_yellow_requires_letter_different_position(self) -> None:
        # Guess "rocky": r is yellow (in word, not pos 0). Gray: o, c, k, y
        candidates = ["ultra", "dwell"]
        fb = [LetterFeedback.YELLOW, LetterFeedback.GRAY, LetterFeedback.GRAY, LetterFeedback.GRAY, LetterFeedback.GRAY]
        result = filter_candidates(candidates, "rocky", fb)
        assert "ultra" in result  # has 'r' at pos 3 (not 0), no o/c/k/y
        assert "dwell" not in result  # no 'r'

    def test_gray_excludes_letter(self) -> None:
        candidates = ["crane", "blown"]
        # Guess "blown": all gray means no b, l, o, w, n in target
        fb = [LetterFeedback.GRAY] * 5
        result = filter_candidates(candidates, "blown", fb)
        assert "crane" not in result  # has 'n'
        assert "blown" not in result  # has all gray letters

    def test_green_plus_yellow_requires_two_of_letter(self) -> None:
        # Guess "aliya" vs target "alack": a(green) l(green) i,y(gray) a(yellow).
        # The trailing yellow 'a' means the answer has a SECOND 'a' beyond the green.
        candidates = ["alack", "allus", "aloft"]
        fb = WordleEnv.compute_feedback("aliya", "alack")
        result = filter_candidates(candidates, "aliya", fb)
        assert "alack" in result  # two a's — consistent
        assert "allus" not in result  # only one 'a' — must be excluded
        assert "aloft" not in result  # only one 'a'


class TestSolvers:
    def test_random_completes_game(self) -> None:
        env = WordleEnv()
        answers = load_answers()[:50]
        state = play_game_random(env, "crane", answers)
        assert state.solved or state.failed
        assert state.turn <= 6

    def test_decent_completes_game(self) -> None:
        env = WordleEnv()
        answers = load_answers()[:50]
        state = play_game_decent(env, "crane", answers)
        assert state.solved or state.failed

    def test_good_completes_game(self) -> None:
        env = WordleEnv()
        answers = load_answers()[:20]
        state = play_game_good(env, answers[0], answers, answers)
        assert state.solved or state.failed


class TestTranscripts:
    def test_generates_examples(self) -> None:
        tokenizer = CharTokenizer()
        examples = generate_examples(tokenizer, n_games=10)
        assert len(examples) > 0

    def test_examples_have_prompt_and_target(self) -> None:
        tokenizer = CharTokenizer()
        examples = generate_examples(tokenizer, n_games=5)
        for ex in examples:
            assert len(ex.prompt_ids) >= 1
            assert len(ex.target_ids) == 5

    def test_prompt_starts_with_bos(self) -> None:
        tokenizer = CharTokenizer()
        examples = generate_examples(tokenizer, n_games=5)
        for ex in examples:
            assert ex.prompt_ids[0] == tokenizer.bos_id
