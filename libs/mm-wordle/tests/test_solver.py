"""Tests for Wordle solvers and transcript generation."""

from mm_tokenizers import CharTokenizer
from mm_wordle.game import LetterFeedback, WordleEnv
from mm_wordle.solver import filter_candidates, play_game_decent, play_game_good, play_game_random
from mm_wordle.transcripts import generate_transcripts
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
    def test_generates_tokens(self) -> None:
        tokenizer = CharTokenizer()
        tokens = generate_transcripts(tokenizer, n_games=10)
        assert len(tokens) > 0

    def test_contains_feedback_tokens(self) -> None:
        tokenizer = CharTokenizer()
        tokens = generate_transcripts(tokenizer, n_games=10)
        assert tokenizer.green_id in tokens
        assert tokenizer.gray_id in tokens

    def test_contains_bos_eos(self) -> None:
        tokenizer = CharTokenizer()
        tokens = generate_transcripts(tokenizer, n_games=5)
        assert tokenizer.bos_id in tokens
        assert tokenizer.eos_id in tokens
