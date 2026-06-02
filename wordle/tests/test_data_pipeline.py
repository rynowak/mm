"""Tests for the pre-training data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import WordleDataset
from mm_tokenizers import CharTokenizer


class TestWordleDataset:
    def test_basic_shape(self) -> None:
        prompts = [torch.tensor([29], dtype=torch.long)]  # [bos]
        targets = [torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)]
        ds = WordleDataset(prompts, targets, pad_id=31)
        input_ids, target_ids, loss_mask = ds[0]
        assert input_ids.shape == (5,)
        assert target_ids.shape == (5,)
        assert loss_mask.shape == (5,)

    def test_loss_mask_on_target_only(self) -> None:
        prompts = [torch.tensor([29, 0, 1, 2, 3, 4, 26, 26, 26, 26, 26, 32], dtype=torch.long)]
        targets = [torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)]
        ds = WordleDataset(prompts, targets, pad_id=31)
        _, _, loss_mask = ds[0]
        assert loss_mask.sum().item() == 5
        assert loss_mask[-5:].sum().item() == 5

    def test_length(self) -> None:
        prompts = [torch.tensor([29], dtype=torch.long) for _ in range(10)]
        targets = [torch.tensor([0, 1, 2, 3, 4], dtype=torch.long) for _ in range(10)]
        ds = WordleDataset(prompts, targets, pad_id=31)
        assert len(ds) == 10


class TestTokenizationFormat:
    def test_bos_in_prompt(self) -> None:
        tokenizer = CharTokenizer()
        bos_id = tokenizer.bos_id
        prompt = [bos_id]
        assert prompt[0] == bos_id

    def test_ascii_filter(self) -> None:
        good = "abcxyz"
        bad = "ABC123!@#âé"
        filtered = [ch for ch in (good + bad) if "a" <= ch <= "z"]
        assert "".join(filtered) == "abcxyz"
