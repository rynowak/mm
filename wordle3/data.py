"""V3 data pipelines.

Pre-train (§5.2): word-only, each word conditioned on the zero-constraint prompt
``[bos] ? ? ? ? ? [sep]``, loss masked to the 5 letters.

SFT (§5.3): behavior cloning on golden games — each turn is a (constraint-state,
guess) example, loss masked to the guess letters. Late turns are oversampled.
"""

from __future__ import annotations

import random as rng
from typing import TYPE_CHECKING

import torch
from mm_wordle import WordleEnv
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from mm_wordle import ConstraintTokenizer, GameState


class WordOnlyDataset(Dataset):
    """(empty-prompt, word) examples for word-only pre-training.

    Each item is ``(input_ids, target_ids, loss_mask)``:
      - input_ids:  (prompt + word)[:-1]
      - target_ids: (prompt + word)[1:]
      - loss_mask:  1 on the 5 word-letter positions, 0 on the prompt.
    """

    def __init__(self, words: list[str], tokenizer: ConstraintTokenizer) -> None:
        prompt = tokenizer.empty_prompt()
        prompt_len = len(prompt)
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for word in words:
            target = [tokenizer.encode_token(ch) for ch in word]
            full = torch.tensor(prompt + target, dtype=torch.long)
            input_ids = full[:-1]
            target_ids = full[1:]
            loss_mask = torch.zeros_like(target_ids, dtype=torch.float32)
            loss_mask[prompt_len - 1 :] = 1.0
            self.examples.append((input_ids, target_ids, loss_mask))
        self.pad_id = tokenizer.pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_padded(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch of (input, target, mask) sequences to equal length."""
    input_ids = pad_sequence([x[0] for x in batch], batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=pad_id)
    loss_masks = pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0.0)
    return input_ids, target_ids, loss_masks


class ConstraintDataset(Dataset):
    """(constraint-state, guess) examples for SFT. Loss masked to the guess letters."""

    def __init__(self, prompts: list[list[int]], targets: list[list[int]], pad_id: int) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for prompt, target in zip(prompts, targets, strict=True):
            full = torch.tensor(prompt + target, dtype=torch.long)
            input_ids = full[:-1]
            target_ids = full[1:]
            loss_mask = torch.zeros_like(target_ids, dtype=torch.float32)
            loss_mask[len(prompt) - 1 :] = 1.0
            self.examples.append((input_ids, target_ids, loss_mask))
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


def golden_examples_from_game(
    state: GameState, tokenizer: ConstraintTokenizer
) -> list[tuple[list[int], list[int], int]]:
    """Replay a completed game into per-turn (prompt_ids, target_ids, turn) examples."""
    env = WordleEnv()
    replay = env.reset(target_word=state.target)
    out: list[tuple[list[int], list[int], int]] = []
    for i, gf in enumerate(state.guesses):
        prompt_ids = tokenizer.encode_game_state(replay)
        target_ids = [tokenizer.encode_token(ch) for ch in gf.guess]
        out.append((prompt_ids, target_ids, i + 1))
        replay, _ = env.step(replay, gf.guess)
    return out


def oversample_late_turns(
    examples: list[tuple[list[int], list[int], int]],
) -> list[tuple[list[int], list[int], int]]:
    """Replicate each turn-N example N times so late turns aren't swamped by openers."""
    result: list[tuple[list[int], list[int], int]] = []
    for ex in examples:
        result.extend([ex] * ex[2])
    rng.shuffle(result)
    return result
