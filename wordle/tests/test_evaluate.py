"""Tests for the evaluation script."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import compute_metrics, generate_guess, play_game
from mm_model import GPT, GPTConfig
from mm_tokenizers import CharTokenizer
from mm_viz import GameReplay
from mm_wordle import WordleEnv

_CPU = torch.device("cpu")


def _make_tiny_model(device: torch.device = _CPU) -> GPT:
    config = GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=50, context_len=128)
    model = GPT(config)
    model.to(device)
    model.eval()
    return model


class TestGenerateGuess:
    def test_returns_5_chars(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        state = env.reset(target_word="crane")

        guess = generate_guess(model, tokenizer, state, torch.device("cpu"))
        assert len(guess) == 5

    def test_returns_only_lowercase_letters(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        state = env.reset(target_word="crane")

        guess = generate_guess(model, tokenizer, state, torch.device("cpu"))
        assert all("a" <= ch <= "z" for ch in guess)

    def test_works_with_game_history(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        state = env.reset(target_word="crane")
        state, _ = env.step(state, "house")

        guess = generate_guess(model, tokenizer, state, torch.device("cpu"))
        assert len(guess) == 5
        assert all("a" <= ch <= "z" for ch in guess)


class TestPlayGame:
    def test_completes_game(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        valid_words = sorted({"crane", "house", "slate", "about", "trace"})

        replay = play_game(model, tokenizer, env, valid_words, torch.device("cpu"), target_word="crane")

        assert isinstance(replay, GameReplay)
        assert replay.target == "crane"
        assert 1 <= replay.turns <= 6
        assert len(replay.guesses) == replay.turns
        assert len(replay.feedback) == replay.turns

    def test_feedback_has_correct_structure(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        valid_words = sorted({"crane"})

        replay = play_game(model, tokenizer, env, valid_words, torch.device("cpu"), target_word="crane")

        for fb in replay.feedback:
            assert len(fb) == 5
            assert all(f in ("green", "yellow", "gray") for f in fb)

    def test_game_ends_after_6_turns(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        valid_words = sorted({"crane"})

        replay = play_game(model, tokenizer, env, valid_words, torch.device("cpu"), target_word="crane")
        assert replay.turns <= 6


class TestComputeMetrics:
    def test_all_wins(self) -> None:
        replays = [
            GameReplay(target="crane", guesses=["crane"], feedback=[["green"] * 5], solved=True, turns=1),
            GameReplay(target="house", guesses=["house"], feedback=[["green"] * 5], solved=True, turns=1),
        ]
        metrics = compute_metrics(replays, "constrained")
        assert metrics.win_rate == 1.0
        assert metrics.wins == 2
        assert metrics.avg_guesses_winners == 1.0

    def test_all_losses(self) -> None:
        replays = [
            GameReplay(
                target="crane",
                guesses=["a"] * 6,
                feedback=[["gray"] * 5] * 6,
                solved=False,
                turns=6,
            ),
        ]
        metrics = compute_metrics(replays, "constrained")
        assert metrics.win_rate == 0.0
        assert metrics.wins == 0
        assert metrics.avg_guesses_winners == 0.0

    def test_mixed_results(self) -> None:
        replays = [
            GameReplay(target="crane", guesses=["crane"], feedback=[["green"] * 5], solved=True, turns=1),
            GameReplay(
                target="house",
                guesses=["x"] * 6,
                feedback=[["gray"] * 5] * 6,
                solved=False,
                turns=6,
            ),
        ]
        metrics = compute_metrics(replays, "constrained")
        assert metrics.win_rate == 0.5
        assert metrics.wins == 1

    def test_guess_distribution(self) -> None:
        replays = [
            GameReplay(
                target="a",
                guesses=["a", "a", "a"],
                feedback=[["gray"] * 5] * 2 + [["green"] * 5],
                solved=True,
                turns=3,
            ),
            GameReplay(target="b", guesses=["b"], feedback=[["green"] * 5], solved=True, turns=1),
        ]
        metrics = compute_metrics(replays, "constrained")
        assert metrics.guess_distribution[1] == 1
        assert metrics.guess_distribution[3] == 1

    def test_first_guess_tracking(self) -> None:
        replays = [
            GameReplay(
                target="crane",
                guesses=["slate", "crane"],
                feedback=[["gray"] * 5, ["green"] * 5],
                solved=True,
                turns=2,
            ),
            GameReplay(
                target="house",
                guesses=["slate", "house"],
                feedback=[["gray"] * 5, ["green"] * 5],
                solved=True,
                turns=2,
            ),
        ]
        metrics = compute_metrics(replays, "constrained")
        assert metrics.first_guesses["slate"] == 2

    def test_empty_replays(self) -> None:
        metrics = compute_metrics([], "constrained")
        assert metrics.win_rate == 0.0
        assert metrics.num_games == 0
