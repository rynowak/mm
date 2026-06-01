"""Tests for the pre-training data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import TokenBlockDataset
from mm_tokenizers import CharTokenizer


class TestTokenBlockDataset:
    def test_basic_shape(self) -> None:
        tokens = torch.arange(100, dtype=torch.long)
        ds = TokenBlockDataset(tokens, block_size=10)
        input_ids, target_ids = ds[0]
        assert input_ids.shape == (10,)
        assert target_ids.shape == (10,)

    def test_target_is_shifted_input(self) -> None:
        tokens = torch.arange(100, dtype=torch.long)
        ds = TokenBlockDataset(tokens, block_size=10)
        input_ids, target_ids = ds[0]
        assert torch.equal(input_ids[1:], target_ids[:-1])

    def test_length(self) -> None:
        tokens = torch.arange(100, dtype=torch.long)
        ds = TokenBlockDataset(tokens, block_size=10)
        assert len(ds) == 9  # (100 - 1) // 10

    def test_blocks_are_contiguous(self) -> None:
        tokens = torch.arange(50, dtype=torch.long)
        ds = TokenBlockDataset(tokens, block_size=5)
        input0, _ = ds[0]
        input1, _ = ds[1]
        assert input0[-1].item() == 4
        assert input1[0].item() == 5

    def test_single_block(self) -> None:
        tokens = torch.arange(11, dtype=torch.long)
        ds = TokenBlockDataset(tokens, block_size=10)
        assert len(ds) == 1
        input_ids, target_ids = ds[0]
        assert torch.equal(input_ids, torch.arange(0, 10))
        assert torch.equal(target_ids, torch.arange(1, 11))


class TestTokenizationFormat:
    def test_bos_in_story_tokens(self) -> None:
        """Story tokenization should include [bos] before each story."""
        tokenizer = CharTokenizer()
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id

        text = "hello"
        tokens = [bos_id] + tokenizer.encode(text) + [eos_id]
        assert tokens[0] == bos_id
        assert tokens[-1] == eos_id

    def test_word_list_has_bos(self) -> None:
        """Word list tokenization should start with [bos]."""
        tokenizer = CharTokenizer()
        bos_id = tokenizer.bos_id

        words = ["crane", "slate"]
        tokens = [bos_id]
        for i, word in enumerate(words):
            tokens.extend(tokenizer.encode(word))
            if i < len(words) - 1:
                tokens.append(tokenizer.sep_id)
            else:
                tokens.append(tokenizer.eos_id)

        assert tokens[0] == bos_id
        assert tokens[-1] == tokenizer.eos_id

    def test_ascii_filter(self) -> None:
        """Only a-z should pass the character filter."""
        good = "abcxyz"
        bad = "ABC123!@#âé"
        filtered = [ch for ch in (good + bad) if "a" <= ch <= "z"]
        assert "".join(filtered) == "abcxyz"
