"""Tests for WordTrie."""

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
