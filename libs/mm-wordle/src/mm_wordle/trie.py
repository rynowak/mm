"""Trie for constrained decoding over a word list."""

from __future__ import annotations

import torch
from torch import Tensor


class WordTrie:
    """Prefix trie with GPU-accelerated mask lookup."""

    def __init__(self) -> None:
        self.children: dict[str, WordTrie] = {}
        self.is_word: bool = False
        self._node_id: int = -1
        self._masks: Tensor | None = None
        self._prefix_to_id: dict[str, int] = {}

    @classmethod
    def from_words(cls, words: list[str]) -> WordTrie:
        root = cls()
        for word in words:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = cls()
                node = node.children[ch]
            node.is_word = True
        return root

    def valid_next_chars(self, prefix: str) -> set[str]:
        node = self
        for ch in prefix:
            if ch not in node.children:
                return set()
            node = node.children[ch]
        return set(node.children.keys())

    def has_prefix(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def is_valid_word(self, word: str) -> bool:
        node = self
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_word

    def build_gpu_masks(self, vocab_size: int, char_to_id: dict[str, int], device: torch.device) -> None:
        """Precompute all trie node masks as a single GPU tensor.

        After calling this, `gpu_mask(prefix)` returns a pre-built mask
        tensor from GPU memory instead of building it in Python.
        """
        nodes: list[WordTrie] = []
        prefix_to_id: dict[str, int] = {}

        def _collect(node: WordTrie, prefix: str) -> None:
            node._node_id = len(nodes)
            prefix_to_id[prefix] = node._node_id
            nodes.append(node)
            for ch in sorted(node.children.keys()):
                _collect(node.children[ch], prefix + ch)

        _collect(self, "")

        masks = torch.full((len(nodes), vocab_size), float("-inf"), device=device)
        for node in nodes:
            for ch in node.children:
                if ch in char_to_id:
                    masks[node._node_id, char_to_id[ch]] = 0.0

        self._masks = masks
        self._prefix_to_id = prefix_to_id

    def gpu_mask(self, prefixes: list[str]) -> Tensor:
        """Look up precomputed masks for a batch of prefixes.

        Returns (batch_size, vocab_size) tensor on GPU.
        """
        assert self._masks is not None, "Call build_gpu_masks first"
        ids = [self._prefix_to_id.get(p, 0) for p in prefixes]
        return self._masks[ids]
