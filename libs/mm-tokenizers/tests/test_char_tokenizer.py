"""Tests for CharTokenizer."""

import pytest
from mm_tokenizers import CharTokenizer


@pytest.fixture
def tok() -> CharTokenizer:
    return CharTokenizer()


# ------------------------------------------------------------------
# Vocabulary basics
# ------------------------------------------------------------------


def test_vocab_size(tok: CharTokenizer) -> None:
    assert tok.vocab_size == 50


# ------------------------------------------------------------------
# Special token ID properties
# ------------------------------------------------------------------


def test_special_token_ids(tok: CharTokenizer) -> None:
    assert tok.green_id == 26
    assert tok.yellow_id == 27
    assert tok.gray_id == 28
    assert tok.bos_id == 29
    assert tok.eos_id == 30
    assert tok.pad_id == 31
    assert tok.sep_id == 32
    assert tok.newline_id == 33


# ------------------------------------------------------------------
# Round-trip: plain text
# ------------------------------------------------------------------


def test_roundtrip_plain_text(tok: CharTokenizer) -> None:
    text = "hello"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_roundtrip_alphabet(tok: CharTokenizer) -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    ids = tok.encode(text)
    assert ids == list(range(26))
    assert tok.decode(ids) == text


# ------------------------------------------------------------------
# Round-trip: text with special tokens
# ------------------------------------------------------------------


def test_roundtrip_with_special_tokens(tok: CharTokenizer) -> None:
    text = "[bos]hello[eos]"
    ids = tok.encode(text)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == text


def test_roundtrip_feedback_tokens(tok: CharTokenizer) -> None:
    text = "[green][yellow][gray]"
    ids = tok.encode(text)
    assert ids == [tok.green_id, tok.yellow_id, tok.gray_id]
    assert tok.decode(ids) == text


def test_roundtrip_mixed(tok: CharTokenizer) -> None:
    text = "[bos]abc[sep]xyz[green][gray][eos]"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_string(tok: CharTokenizer) -> None:
    assert tok.encode("") == []


def test_decode_empty_list(tok: CharTokenizer) -> None:
    assert tok.decode([]) == ""


def test_unknown_character_raises(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown token"):
        tok.encode("A")  # uppercase not in vocab


def test_unknown_digit_raises(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown token"):
        tok.encode("1")


def test_unknown_token_id_raises(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown token ID"):
        tok.decode([999])


def test_unknown_bracket_token_raises(tok: CharTokenizer) -> None:
    with pytest.raises(ValueError, match="Unknown token"):
        tok.encode("[unknown]")
