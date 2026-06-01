"""Character-level tokenizer with a fixed ~50-token vocabulary."""

from __future__ import annotations

import re

# Vocabulary layout (50 tokens total):
#   0-25:  lowercase letters a-z
#  26-28:  feedback tokens [green] [yellow] [gray]
#  29-33:  structural tokens [bos] [eos] [pad] [sep] [newline]
#  34-49:  reserved for future use

_LETTERS = list("abcdefghijklmnopqrstuvwxyz")  # 0-25

_SPECIAL_TOKENS: list[str] = [
    "[green]",  # 26
    "[yellow]",  # 27
    "[gray]",  # 28
    "[bos]",  # 29
    "[eos]",  # 30
    "[pad]",  # 31
    "[sep]",  # 32
    "[newline]",  # 33
]

_RESERVED_COUNT = 16  # slots 34-49

# Build forward and reverse lookup tables once at module level.
_TOKEN_TO_ID: dict[str, int] = {}
_ID_TO_TOKEN: dict[int, str] = {}

for _i, _ch in enumerate(_LETTERS):
    _TOKEN_TO_ID[_ch] = _i
    _ID_TO_TOKEN[_i] = _ch

for _i, _tok in enumerate(_SPECIAL_TOKENS, start=len(_LETTERS)):
    _TOKEN_TO_ID[_tok] = _i
    _ID_TO_TOKEN[_i] = _tok

_VOCAB_SIZE = len(_LETTERS) + len(_SPECIAL_TOKENS) + _RESERVED_COUNT  # 50

# Regex that greedily matches bracketed special tokens first, then single chars.
_TOKENIZE_RE = re.compile(r"\[[a-z]+\]|.")


class CharTokenizer:
    """Fixed-vocabulary character-level tokenizer.

    Vocabulary (~50 tokens):
      * 26 lowercase letters (a-z)
      * 3 feedback tokens: ``[green]``, ``[yellow]``, ``[gray]``
      * 5 structural tokens: ``[bos]``, ``[eos]``, ``[pad]``, ``[sep]``, ``[newline]``
      * 16 reserved slots for future use
    """

    # ------------------------------------------------------------------
    # Vocabulary introspection
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including reserved slots."""
        return _VOCAB_SIZE

    # ------------------------------------------------------------------
    # Special-token ID properties
    # ------------------------------------------------------------------

    @property
    def bos_id(self) -> int:
        return _TOKEN_TO_ID["[bos]"]

    @property
    def eos_id(self) -> int:
        return _TOKEN_TO_ID["[eos]"]

    @property
    def pad_id(self) -> int:
        return _TOKEN_TO_ID["[pad]"]

    @property
    def sep_id(self) -> int:
        return _TOKEN_TO_ID["[sep]"]

    @property
    def newline_id(self) -> int:
        return _TOKEN_TO_ID["[newline]"]

    @property
    def green_id(self) -> int:
        return _TOKEN_TO_ID["[green]"]

    @property
    def yellow_id(self) -> int:
        return _TOKEN_TO_ID["[yellow]"]

    @property
    def gray_id(self) -> int:
        return _TOKEN_TO_ID["[gray]"]

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of token IDs.

        Special tokens must appear in bracket notation (e.g. ``[bos]``).
        Raises :class:`ValueError` for any character or bracket token not
        in the vocabulary.
        """
        if not text:
            return []

        ids: list[int] = []
        for token in _TOKENIZE_RE.findall(text):
            if token not in _TOKEN_TO_ID:
                raise ValueError(f"Unknown token: {token!r}")
            ids.append(_TOKEN_TO_ID[token])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        parts: list[str] = []
        for tid in token_ids:
            if tid not in _ID_TO_TOKEN:
                raise ValueError(f"Unknown token ID: {tid}")
            parts.append(_ID_TO_TOKEN[tid])
        return "".join(parts)
