"""Tests for the RL fine-tuning script."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune import (
    compute_guess_log_probs,
    create_reference_model,
    load_pretrained_model,
    sample_constrained,
    sample_unconstrained,
)
from mm_model import GPT, GPTConfig, save_checkpoint
from mm_tokenizers import CharTokenizer
from mm_wordle import WordTrie

_CPU = torch.device("cpu")


def _make_tiny_model(device: torch.device = _CPU) -> GPT:
    config = GPTConfig(n_layers=2, n_heads=2, embed_dim=32, vocab_size=50, context_len=128)
    model = GPT(config)
    model.to(device)
    return model


class TestCreateReferenceModel:
    def test_frozen_params(self) -> None:
        model = _make_tiny_model()
        ref = create_reference_model(model)
        for param in ref.parameters():
            assert not param.requires_grad

    def test_produces_same_output(self) -> None:
        model = _make_tiny_model()
        model.eval()
        ref = create_reference_model(model)

        x = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            out_model, _ = model(x)
            out_ref, _ = ref(x)
        assert torch.allclose(out_model, out_ref)

    def test_is_independent_copy(self) -> None:
        model = _make_tiny_model()
        ref = create_reference_model(model)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        x = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            out_model, _ = model(x)
            out_ref, _ = ref(x)
        assert not torch.allclose(out_model, out_ref)


class TestSampleConstrained:
    def test_returns_valid_word(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        words = ["crane", "house", "slate"]
        trie = WordTrie.from_words(words)
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        results = sample_constrained(model, game_state_ids, trie, tokenizer, _CPU)
        assert len(results) == 1
        word, ids = results[0]
        assert trie.is_valid_word(word)
        assert ids.shape == (5,)

    def test_multiple_samples(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        words = ["crane", "house", "slate"]
        trie = WordTrie.from_words(words)
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        results = sample_constrained(model, game_state_ids, trie, tokenizer, _CPU, n_samples=5)
        assert len(results) == 5
        for word, _ids in results:
            assert trie.is_valid_word(word)

    def test_restores_training_mode(self) -> None:
        model = _make_tiny_model()
        model.train()
        tokenizer = CharTokenizer()
        trie = WordTrie.from_words(["crane"])
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        sample_constrained(model, game_state_ids, trie, tokenizer, _CPU)
        assert model.training

    def test_generates_5_char_words(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        trie = WordTrie.from_words(["crane", "house", "slate", "about", "train"])
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        results = sample_constrained(model, game_state_ids, trie, tokenizer, _CPU, n_samples=10)
        for word, _ids in results:
            assert len(word) == 5


class TestSampleUnconstrained:
    def test_returns_word_and_ids(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        results = sample_unconstrained(model, game_state_ids, _CPU, tokenizer)
        assert len(results) == 1
        word, ids = results[0]
        assert isinstance(word, str)
        assert ids.shape == (5,)

    def test_multiple_samples(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()
        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)

        results = sample_unconstrained(model, game_state_ids, _CPU, tokenizer, n_samples=3)
        assert len(results) == 3


class TestComputeGuessLogProbs:
    def test_returns_5_log_probs(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()

        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)
        word_ids = torch.tensor(tokenizer.encode("crane"), dtype=torch.long)

        lp = compute_guess_log_probs(model, game_state_ids, word_ids)
        assert lp.shape == (5,)

    def test_log_probs_are_negative(self) -> None:
        model = _make_tiny_model()
        tokenizer = CharTokenizer()

        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)
        word_ids = torch.tensor(tokenizer.encode("crane"), dtype=torch.long)

        lp = compute_guess_log_probs(model, game_state_ids, word_ids)
        assert (lp <= 0).all()

    def test_has_gradients(self) -> None:
        model = _make_tiny_model()
        model.train()
        tokenizer = CharTokenizer()

        game_state_ids = torch.tensor(tokenizer.encode("[bos]"), dtype=torch.long)
        word_ids = torch.tensor(tokenizer.encode("crane"), dtype=torch.long)

        lp = compute_guess_log_probs(model, game_state_ids, word_ids)
        loss = lp.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


class TestLoadPretrainedModel:
    def test_loads_from_checkpoint(self, tmp_path: Path) -> None:
        model = _make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "model.pt"
        save_checkpoint(ckpt_path, model, optimizer, step=100, config=model.config)

        loaded = load_pretrained_model(ckpt_path, _CPU)
        assert isinstance(loaded, GPT)

        x = torch.randint(0, 50, (1, 10))
        model.eval()
        loaded.eval()
        with torch.no_grad():
            out_orig, _ = model(x)
            out_loaded, _ = loaded(x)
        assert torch.allclose(out_orig, out_loaded)
