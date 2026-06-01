"""Tests for BPETokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm_tokenizers import BPETokenizer

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Training basics
# ------------------------------------------------------------------


class TestTrain:
    def test_vocab_size_increases_to_target(self) -> None:
        tok = BPETokenizer()
        assert tok.vocab_size == 256
        tok.train("the cat sat on the mat " * 50, target_vocab_size=266)
        assert tok.vocab_size == 266

    def test_known_merge_repeated_chars(self) -> None:
        """Training on 'aaaa' should merge 'a','a' -> 'aa' first."""
        tok = BPETokenizer()
        tok.train("aaaa", target_vocab_size=257)
        assert len(tok.merges) == 1
        a_byte = ord("a")
        assert tok.merges[0] == (a_byte, a_byte)
        # The new token should be b'aa'
        assert tok.vocab[256] == b"aa"

    def test_train_on_real_english(self) -> None:
        text = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. ") * 100
        tok = BPETokenizer()
        tok.train(text, target_vocab_size=300)
        assert tok.vocab_size == 300
        # Common pairs like "th", "he", " t" should be merged
        merged_bytes = {tok.vocab[256 + i] for i in range(len(tok.merges))}
        # At minimum, some multi-byte tokens should exist
        assert any(len(b) >= 2 for b in merged_bytes)


# ------------------------------------------------------------------
# Encode / decode round-trip
# ------------------------------------------------------------------


class TestEncodeDecode:
    def test_roundtrip_preserves_text(self) -> None:
        tok = BPETokenizer()
        tok.train("hello world " * 100, target_vocab_size=270)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_encode_fewer_tokens_than_bytes(self) -> None:
        """After training, encoding should produce fewer tokens than raw bytes."""
        tok = BPETokenizer()
        text = "abcabc " * 200
        tok.train(text, target_vocab_size=270)
        encoded = tok.encode("abcabc abcabc")
        raw_bytes = len(b"abcabc abcabc")
        assert len(encoded) < raw_bytes

    def test_empty_string(self) -> None:
        tok = BPETokenizer()
        assert tok.encode("") == []

    def test_roundtrip_untrained(self) -> None:
        """Without training, encode/decode is byte-level identity."""
        tok = BPETokenizer()
        text = "hello"
        ids = tok.encode(text)
        assert ids == list(text.encode("utf-8"))
        assert tok.decode(ids) == text

    def test_roundtrip_unicode(self) -> None:
        tok = BPETokenizer()
        tok.train("café " * 100, target_vocab_size=270)
        text = "café"
        assert tok.decode(tok.encode(text)) == text


# ------------------------------------------------------------------
# Save / load round-trip
# ------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        tok = BPETokenizer()
        tok.train("the cat sat on the mat " * 50, target_vocab_size=280)
        path = str(tmp_path / "bpe.json")
        tok.save(path)

        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.merges == tok.merges

        # Encoding should produce same results
        text = "the cat sat"
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.decode(loaded.encode(text)) == text
