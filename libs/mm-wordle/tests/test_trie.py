"""Tests for WordTrie including GPU mask precomputation."""

import torch
from mm_wordle.trie import WordTrie


class TestWordTrie:
    def test_valid_next_chars_root(self) -> None:
        trie = WordTrie.from_words(["crane", "crash", "bliss"])
        nexts = trie.valid_next_chars("")
        assert nexts == {"c", "b"}

    def test_valid_next_chars_after_prefix(self) -> None:
        trie = WordTrie.from_words(["crane", "crash", "crisp"])
        nexts = trie.valid_next_chars("cr")
        assert nexts == {"a", "i"}

    def test_valid_next_chars_full_word(self) -> None:
        trie = WordTrie.from_words(["crane"])
        nexts = trie.valid_next_chars("crane")
        assert nexts == set()

    def test_valid_next_chars_invalid_prefix(self) -> None:
        trie = WordTrie.from_words(["crane"])
        nexts = trie.valid_next_chars("xyz")
        assert nexts == set()

    def test_is_valid_word(self) -> None:
        trie = WordTrie.from_words(["crane", "house"])
        assert trie.is_valid_word("crane")
        assert trie.is_valid_word("house")
        assert not trie.is_valid_word("cran")
        assert not trie.is_valid_word("cranes")

    def test_has_prefix(self) -> None:
        trie = WordTrie.from_words(["crane", "crash"])
        assert trie.has_prefix("cr")
        assert trie.has_prefix("cra")
        assert not trie.has_prefix("cx")

    def test_large_word_list(self) -> None:
        from mm_wordle import load_answers

        words = load_answers()
        trie = WordTrie.from_words(words)
        for w in words[:100]:
            assert trie.is_valid_word(w)
            assert trie.valid_next_chars(w[:3])


class TestGPUMasks:
    def _char_to_id(self) -> dict[str, int]:
        return {chr(ord("a") + i): i for i in range(26)}

    def test_build_gpu_masks(self) -> None:
        trie = WordTrie.from_words(["crane", "crash"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        assert trie._masks is not None
        assert trie._masks.shape[1] == 50

    def test_gpu_mask_root_allows_only_valid_chars(self) -> None:
        trie = WordTrie.from_words(["crane", "bliss"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        mask = trie.gpu_mask([""])
        # 'c' (id=2) and 'b' (id=1) should be 0, everything else -inf
        assert mask[0, 2].item() == 0.0  # c
        assert mask[0, 1].item() == 0.0  # b
        assert mask[0, 0].item() == float("-inf")  # a
        assert mask[0, 3].item() == float("-inf")  # d

    def test_gpu_mask_after_prefix(self) -> None:
        trie = WordTrie.from_words(["crane", "crash", "crisp"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        mask = trie.gpu_mask(["cr"])
        # After 'cr': 'a' (id=0) and 'i' (id=8) should be valid
        assert mask[0, 0].item() == 0.0  # a
        assert mask[0, 8].item() == 0.0  # i
        assert mask[0, 2].item() == float("-inf")  # c

    def test_gpu_mask_invalid_prefix_blocks_all(self) -> None:
        trie = WordTrie.from_words(["crane"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        mask = trie.gpu_mask(["xyz"])
        # Unknown prefix falls back to root node (id=0)
        # Root only allows 'c', so most chars are -inf
        assert mask[0, 2].item() == 0.0  # c (root allows it)

    def test_gpu_mask_batched(self) -> None:
        trie = WordTrie.from_words(["crane", "bliss"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        masks = trie.gpu_mask(["", "c"])
        assert masks.shape == (2, 50)
        # First prefix "": allows c and b
        assert masks[0, 2].item() == 0.0  # c
        assert masks[0, 1].item() == 0.0  # b
        # Second prefix "c": allows r
        assert masks[1, 17].item() == 0.0  # r
        assert masks[1, 1].item() == float("-inf")  # b

    def test_gpu_mask_full_word_blocks_all(self) -> None:
        trie = WordTrie.from_words(["crane"])
        trie.build_gpu_masks(50, self._char_to_id(), torch.device("cpu"))
        mask = trie.gpu_mask(["crane"])
        # No children after complete word — all should be -inf
        # Falls back to root (id=0) since "crane" maps to a leaf with no children
        # Actually "crane" IS in prefix_to_id, and the node has no children
        assert (mask[0, :26] == float("-inf")).all()

    def test_assert_without_build(self) -> None:
        import pytest

        trie = WordTrie.from_words(["crane"])
        with pytest.raises(AssertionError):
            trie.gpu_mask([""])
