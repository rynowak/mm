"""Data pipeline for pre-training on Wordle game transcripts.

Each training example is a partial game state (prompt) + the next guess (target).
Loss is computed only on the 5 target letter tokens — never on feedback or separator
tokens. This matches RL, where the model only generates letters.

See docs/game-format.md for the token format specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mm_wordle.transcripts import generate_examples
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from mm_tokenizers import CharTokenizer


class WordleDataset(Dataset):
    """Dataset of (prompt, target) pairs for Wordle pre-training.

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

    def __getitem__(self, index):
        return self.examples[index]


def collate_wordle(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int):
    """Collate function that pads to the longest example in the batch."""
    input_ids = [x[0] for x in batch]
    target_ids = [x[1] for x in batch]
    loss_masks = [x[2] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)
    loss_masks = pad_sequence(loss_masks, batch_first=True, padding_value=0.0)

    return input_ids, target_ids, loss_masks


def load_pretrain_data(
    config: dict,
    tokenizer: CharTokenizer,
) -> tuple[WordleDataset, WordleDataset]:
    """Load and prepare pre-training data.

    Generates Wordle game transcripts at mixed skill levels and splits
    into per-turn examples. Each example: prompt (game state) + target
    (5 letter tokens for the next guess).

    Returns (train_dataset, val_dataset).
    """
    val_fraction = config["data"]["val_fraction"]
    n_games = config["data"].get("transcript_games", 20000)

    print(f"Generating {n_games} Wordle game transcripts...")
    examples = generate_examples(tokenizer, n_games=n_games)
    print(f"  Generated {len(examples)} per-turn examples from {n_games} games")

    prompts = [torch.tensor(ex.prompt_ids, dtype=torch.long) for ex in examples]
    targets = [torch.tensor(ex.target_ids, dtype=torch.long) for ex in examples]

    n_val = int(len(examples) * val_fraction)
    n_train = len(examples) - n_val

    train_ds = WordleDataset(prompts[:n_train], targets[:n_train], tokenizer.pad_id)
    val_ds = WordleDataset(prompts[n_train:], targets[n_train:], tokenizer.pad_id)

    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    return train_ds, val_ds
