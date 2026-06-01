"""Data pipeline for pre-training on TinyStories + Wordle word lists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from datasets import load_dataset
from mm_wordle import all_valid_words
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from mm_tokenizers import CharTokenizer


class TokenBlockDataset(Dataset):
    """A PyTorch Dataset that serves fixed-length blocks from a flat token tensor.

    Each item returns (input_ids, target_ids) where target_ids is input_ids
    shifted by one position (standard LM next-token prediction).
    """

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        self.tokens = tokens
        self.block_size = block_size
        # We need block_size + 1 tokens per example (input + shifted target)
        self.n_examples = (len(tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids


def _tokenize_tinystories(
    config: dict,
    tokenizer: CharTokenizer,
) -> list[int]:
    """Load TinyStories from HuggingFace and character-tokenize it.

    Takes a subset to keep total size manageable for small models.
    Character-level tokenization produces ~4-5x more tokens than BPE,
    so we limit the number of stories loaded.
    """
    dataset_name = config["data"]["dataset"]
    ds = load_dataset(dataset_name, split="train")

    # Take a subset -- ~20k stories gives roughly 5M+ character tokens
    n_stories = 20_000
    if len(ds) > n_stories:
        ds = ds.select(range(n_stories))

    tokens: list[int] = []
    for example in ds:
        text = example["text"].strip().lower()
        if not text:
            continue

        # Filter to only characters in our vocabulary (letters + whitespace mapped to sep)
        filtered: list[str] = []
        for ch in text:
            if ch == "\n":
                filtered.append("[newline]")
            elif ch == " ":
                filtered.append("[sep]")
            elif "a" <= ch <= "z":
                filtered.append(ch)
            # Skip characters not in our vocabulary (punctuation, digits, etc.)

        if not filtered:
            continue

        encoded = tokenizer.encode("".join(filtered))
        tokens.extend(encoded)
        tokens.append(tokenizer.eos_id)

    return tokens


def _tokenize_word_lists(
    config: dict,
    tokenizer: CharTokenizer,
) -> list[int]:
    """Tokenize Wordle word lists with repetition for balancing.

    Each word is placed on its own line with [sep] between words.
    The entire list is repeated word_list_repeats times.
    """
    repeats = config["data"]["word_list_repeats"]
    words = sorted(all_valid_words())

    # Build one pass of the word list: word1[sep]word2[sep]...wordN[eos]
    single_pass: list[int] = []
    for i, word in enumerate(words):
        single_pass.extend(tokenizer.encode(word))
        if i < len(words) - 1:
            single_pass.append(tokenizer.sep_id)
        else:
            single_pass.append(tokenizer.eos_id)

    # Repeat
    tokens: list[int] = single_pass * repeats
    return tokens


def load_pretrain_data(
    config: dict,
    tokenizer: CharTokenizer,
) -> tuple[TokenBlockDataset, TokenBlockDataset]:
    """Load and prepare pre-training data.

    1. Load TinyStories from HuggingFace datasets
    2. Load Wordle word lists from mm_wordle
    3. Character-level tokenize everything
    4. Concatenate all tokens into one long sequence
    5. Chunk into fixed-length blocks (context_len + 1 for input/target shift)
    6. Split into train/val

    Returns (train_dataset, val_dataset) as PyTorch Datasets.
    """
    block_size = config["model"]["context_len"]
    val_fraction = config["data"]["val_fraction"]

    print("Tokenizing TinyStories...")
    story_tokens = _tokenize_tinystories(config, tokenizer)
    print(f"  TinyStories: {len(story_tokens):,} tokens")

    print("Tokenizing word lists...")
    word_tokens = _tokenize_word_lists(config, tokenizer)
    print(f"  Word lists: {len(word_tokens):,} tokens")

    # Concatenate all tokens
    all_tokens = story_tokens + word_tokens
    print(f"  Total: {len(all_tokens):,} tokens")

    # Convert to tensor
    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long)

    # Split into train/val
    n_val = int(len(all_tokens_t) * val_fraction)
    n_train = len(all_tokens_t) - n_val

    train_tokens = all_tokens_t[:n_train]
    val_tokens = all_tokens_t[n_train:]

    train_dataset = TokenBlockDataset(train_tokens, block_size)
    val_dataset = TokenBlockDataset(val_tokens, block_size)

    print(f"  Train: {len(train_dataset):,} blocks of {block_size}")
    print(f"  Val:   {len(val_dataset):,} blocks of {block_size}")

    return train_dataset, val_dataset
