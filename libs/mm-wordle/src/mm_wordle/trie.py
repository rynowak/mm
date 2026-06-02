"""Trie for constrained decoding over a word list."""

from __future__ import annotations


class WordTrie:
    """Prefix trie built from a word list.

    Used for constrained decoding: at each character position, returns
    the set of valid next characters given the prefix so far.
    """

    def __init__(self) -> None:
        self.children: dict[str, WordTrie] = {}
        self.is_word: bool = False

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
        """Return the set of characters that can follow this prefix."""
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
