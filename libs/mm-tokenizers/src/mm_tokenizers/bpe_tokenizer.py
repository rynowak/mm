"""Byte Pair Encoding tokenizer built from scratch.

This is a learning exercise -- not used for Wordle, but built to understand
how subword tokenization works in real LLMs.

The algorithm:
1. Start with a byte-level vocabulary (256 entries, one per byte value).
2. Repeatedly find the most frequent adjacent pair of tokens in the data,
   merge them into a new token, and record the merge rule.
3. Stop when the vocabulary reaches the target size.

Encoding applies the learned merge rules in order. Decoding concatenates
the byte strings for each token ID and decodes as UTF-8.
"""

from __future__ import annotations

import json


def _get_pair_counts(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Count all adjacent pairs in a list of token IDs."""
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of *pair* in *token_ids* with *new_id*."""
    result: list[int] = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(token_ids[i])
            i += 1
    return result


class BPETokenizer:
    """Byte Pair Encoding tokenizer built from scratch.

    This is a learning exercise -- not used for Wordle, but built to understand
    how subword tokenization works in real LLMs.
    """

    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []  # ordered merge rules
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    def train(self, text: str, target_vocab_size: int = 256 + 100) -> None:
        """Train BPE merges from text.

        Algorithm:
        1. Start with byte-level vocabulary (256 entries).
        2. Count all adjacent token pairs in the training data.
        3. Find the most frequent pair.
        4. Merge that pair into a new token.
        5. Update the data and pair counts.
        6. Repeat until target_vocab_size is reached.

        Store the merge rules in order (needed for encoding).
        """
        if target_vocab_size <= 256:
            return

        # Reset to base vocabulary
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Convert text to byte-level token IDs
        token_ids = list(text.encode("utf-8"))

        num_merges = target_vocab_size - 256
        for _ in range(num_merges):
            counts = _get_pair_counts(token_ids)
            if not counts:
                break  # no more pairs to merge

            # Find the most frequent pair
            best_pair = max(counts, key=counts.__getitem__)

            # Create new token ID
            new_id = 256 + len(self.merges)

            # Record the merge
            self.merges.append(best_pair)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to the training data
            token_ids = _merge_pair(token_ids, best_pair, new_id)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        1. Convert text to bytes (each byte is a token ID 0-255).
        2. Apply merge rules in order: for each merge ``(a, b) -> c``,
           scan through the sequence and replace adjacent ``(a, b)`` with ``c``.
        3. Return the final token IDs.
        """
        if not text:
            return []

        token_ids = list(text.encode("utf-8"))

        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            token_ids = _merge_pair(token_ids, (a, b), new_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text.

        Look up bytes for each token ID in vocab, concatenate, decode as UTF-8.
        """
        raw = b"".join(self.vocab[tid] for tid in token_ids)
        return raw.decode("utf-8")

    def save(self, path: str) -> None:
        """Save trained tokenizer (merges + vocab) to a JSON file."""
        data = {
            "merges": self.merges,
            # Store vocab values as lists of ints (bytes) for JSON compatibility
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load a trained tokenizer from file."""
        with open(path) as f:
            data = json.load(f)

        tok = cls()
        tok.merges = [tuple(pair) for pair in data["merges"]]  # type: ignore[misc]
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        return tok
