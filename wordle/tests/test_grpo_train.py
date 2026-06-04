"""Tests for GRPO training module."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from grpo_train import collect_game_experience
from mm_model import GPT, GPTConfig
from mm_tokenizers import CharTokenizer
from mm_wordle import WordleEnv, WordTrie

_CPU = torch.device("cpu")
_CHAR_TO_ID = {chr(ord("a") + i): i for i in range(26)}


def _make_trie(words: list[str]) -> WordTrie:
    trie = WordTrie.from_words(words)
    trie.build_gpu_masks(50, _CHAR_TO_ID, _CPU)
    return trie


def _make_tiny_model() -> GPT:
    config = GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=50, context_len=128)
    model = GPT(config)
    return model


class TestCollectGameExperience:
    def test_replay_turns_match_guesses(self) -> None:
        """Replay.turns should equal len(replay.guesses)."""
        model = _make_tiny_model()
        ref_model = _make_tiny_model()
        ref_model.eval()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        words = ["crane", "house", "slate", "about", "train"]
        trie = _make_trie(words)

        torch.manual_seed(42)
        _, replay, _, _ = collect_game_experience(
            model=model,
            ref_model=ref_model,
            env=env,
            target_word="crane",
            tokenizer=tokenizer,
            answers=words,
            trie=trie,
            device=_CPU,
            group_size=2,
            constrained=True,
            max_turns=6,
        )
        assert replay.turns == len(replay.guesses)

    def test_replay_turns_with_initial_state(self) -> None:
        """When starting from an initial state (opener), turns should not double-count."""
        model = _make_tiny_model()
        ref_model = _make_tiny_model()
        ref_model.eval()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        words = ["crane", "house", "slate", "about", "train"]
        trie = _make_trie(words)

        # Simulate opener playing 2 turns
        init_state = env.reset(target_word="crane")
        init_state, _ = env.step(init_state, "slate")
        init_state, _ = env.step(init_state, "house")
        assert init_state.turn == 2

        torch.manual_seed(42)
        _, replay, _, _ = collect_game_experience(
            model=model,
            ref_model=ref_model,
            env=env,
            target_word="crane",
            tokenizer=tokenizer,
            answers=words,
            trie=trie,
            device=_CPU,
            group_size=2,
            constrained=True,
            max_turns=4,
            initial_state=init_state,
            initial_candidates=words,
        )

        # replay.guesses only has the model's guesses (not opener)
        model_guesses = len(replay.guesses)
        # replay.turns = state.turn which includes opener turns
        assert replay.turns == model_guesses + 2

    def test_combined_replay_turns_no_double_count(self) -> None:
        """Simulates what rl_steps.py does: prepend opener guesses to replay."""
        model = _make_tiny_model()
        ref_model = _make_tiny_model()
        ref_model.eval()
        tokenizer = CharTokenizer()
        env = WordleEnv()
        words = ["crane", "house", "slate", "about", "train"]
        trie = _make_trie(words)

        # Simulate opener
        init_state = env.reset(target_word="crane")
        opener_guesses = []
        opener_feedback = []
        for guess_word in ["slate", "house"]:
            init_state, _ = env.step(init_state, guess_word)
            fb = init_state.guesses[-1].feedback
            opener_guesses.append(guess_word)
            opener_feedback.append([f.value for f in fb])

        torch.manual_seed(42)
        _, replay, _, _ = collect_game_experience(
            model=model,
            ref_model=ref_model,
            env=env,
            target_word="crane",
            tokenizer=tokenizer,
            answers=words,
            trie=trie,
            device=_CPU,
            group_size=2,
            constrained=True,
            max_turns=4,
            initial_state=init_state,
            initial_candidates=words,
        )

        # Combine like rl_steps.py does (after fix)
        combined_guesses = opener_guesses + replay.guesses
        combined_turns = replay.turns  # NOT len(opener_guesses) + replay.turns

        assert combined_turns == len(combined_guesses)
        assert combined_turns <= 6
