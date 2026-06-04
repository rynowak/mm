"""Data pipeline for pre-training on Wordle game transcripts.

Two modes:
- Character-level: predicts next 5 characters (original autoregressive approach)
- Word-level: predicts word index from the answer list (classifier approach)

See docs/game-format.md for the token format specification.
"""

from __future__ import annotations

import random as rng
from typing import TYPE_CHECKING

import torch
from mm_wordle.transcripts import generate_examples
from mm_wordle.words import load_answers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from mm_tokenizers import CharTokenizer


_ANSWERS = load_answers()
_ANSWER_SET = set(_ANSWERS)
_WORD_TO_IDX = {w: i for i, w in enumerate(_ANSWERS)}


def word_to_idx(word: str) -> int:
    return _WORD_TO_IDX[word]


def idx_to_word(idx: int) -> str:
    return _ANSWERS[idx]


def num_words() -> int:
    return len(_ANSWERS)


class WordClassifierDataset(Dataset):
    """Dataset of (prompt, word_index) pairs for word-level pre-training.

    Each item returns (input_ids, target_word_idx):
      - input_ids: the game state prompt tokens
      - target_word_idx: index into the answer word list
    """

    def __init__(self, prompts: list[torch.Tensor], targets: list[int], pad_id: int) -> None:
        self.prompts = prompts
        self.targets = targets
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.prompts[index], self.targets[index]


def collate_word_classifier(batch: list[tuple[torch.Tensor, int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [x[0] for x in batch]
    targets = torch.tensor([x[1] for x in batch], dtype=torch.long)
    prompts_padded = pad_sequence(prompts, batch_first=True, padding_value=pad_id)
    return prompts_padded, targets


class WordleDataset(Dataset):
    """Dataset of (prompt, target) pairs for character-level pre-training.

    Each item returns (input_ids, target_ids, loss_mask):
      - input_ids: prompt + target[:-1] (the full sequence the model sees)
      - target_ids: prompt[1:] + target (the shifted prediction targets)
      - loss_mask: 1 for target letter positions, 0 elsewhere
    """

    def __init__(self, prompts: list[torch.Tensor], targets: list[torch.Tensor], pad_id: int) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for prompt, target in zip(prompts, targets, strict=True):
            full_seq = torch.cat([prompt, target])
            input_ids = full_seq[:-1]
            target_ids = full_seq[1:]

            loss_mask = torch.zeros_like(target_ids)
            prompt_len = len(prompt)
            loss_mask[prompt_len - 1 :] = 1.0

            self.examples.append((input_ids, target_ids, loss_mask))

        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_wordle(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that pads to the longest example in the batch."""
    input_ids = [x[0] for x in batch]
    target_ids = [x[1] for x in batch]
    loss_masks = [x[2] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
    loss_masks = pad_sequence(loss_masks, batch_first=True, padding_value=0.0)

    return input_ids, target_ids, loss_masks


def _oversample_late_turns(
    examples: list,
    sep_id: int,
) -> list:
    """Oversample later turns so the model trains more on long-context examples."""
    by_turn: dict[int, list] = {}
    for ex in examples:
        turn = sum(1 for t in ex.prompt_ids if t == sep_id) + 1
        by_turn.setdefault(turn, []).append(ex)

    weights = {t: t for t in by_turn}

    result: list = []
    for turn, exs in by_turn.items():
        w = weights[turn]
        result.extend(exs * w)

    rng.shuffle(result)
    return result


def load_pretrain_data(
    config: dict,
    tokenizer: CharTokenizer,
) -> tuple[WordleDataset, WordleDataset]:
    """Load character-level pre-training data."""
    val_fraction = config["data"]["val_fraction"]
    n_games = config["data"].get("transcript_games", 20000)

    print(f"Generating {n_games} Wordle game transcripts...")
    examples = generate_examples(tokenizer, n_games=n_games)
    print(f"  Generated {len(examples)} per-turn examples from {n_games} games")

    sep_id = tokenizer.encode("[sep]")[0]
    examples = _oversample_late_turns(examples, sep_id)
    print(f"  After oversampling late turns: {len(examples)} examples")

    prompts = [torch.tensor(ex.prompt_ids, dtype=torch.long) for ex in examples]
    targets = [torch.tensor(ex.target_ids, dtype=torch.long) for ex in examples]

    n_val = int(len(examples) * val_fraction)
    n_train = len(examples) - n_val

    train_ds = WordleDataset(prompts[:n_train], targets[:n_train], tokenizer.pad_id)
    val_ds = WordleDataset(prompts[n_train:], targets[n_train:], tokenizer.pad_id)

    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    return train_ds, val_ds


def load_word_classifier_data(
    config: dict,
    tokenizer: CharTokenizer,
) -> tuple[WordClassifierDataset, WordClassifierDataset]:
    """Load word-level classifier pre-training data."""
    val_fraction = config["data"]["val_fraction"]
    n_games = config["data"].get("transcript_games", 20000)

    print(f"Generating {n_games} Wordle game transcripts...")
    examples = generate_examples(tokenizer, n_games=n_games)
    print(f"  Generated {len(examples)} per-turn examples from {n_games} games")

    # Filter to only answer-word targets
    answer_examples = [ex for ex in examples if tokenizer.decode(ex.target_ids) in _ANSWER_SET]
    print(f"  Filtered to answer-word targets: {len(answer_examples)} of {len(examples)}")

    sep_id = tokenizer.encode("[sep]")[0]
    answer_examples = _oversample_late_turns(answer_examples, sep_id)
    print(f"  After oversampling late turns: {len(answer_examples)} examples")

    prompts = [torch.tensor(ex.prompt_ids, dtype=torch.long) for ex in answer_examples]
    targets = [word_to_idx(tokenizer.decode(ex.target_ids)) for ex in answer_examples]

    n_val = int(len(answer_examples) * val_fraction)
    n_train = len(answer_examples) - n_val

    train_ds = WordClassifierDataset(prompts[:n_train], targets[:n_train], tokenizer.pad_id)
    val_ds = WordClassifierDataset(prompts[n_train:], targets[n_train:], tokenizer.pad_id)

    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")
    print(f"  Output classes: {num_words()} words")

    return train_ds, val_ds
