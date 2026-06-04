"""V2 tokenizer: dense constraint-state encoding for Wordle.

Vocabulary (~265 tokens):
  0-25:   plain letters a-z (for generating guesses)
  26:     ? (unknown position in green state)
  27-52:  green tokens a-green through z-green
  53-182: yellow tokens a-yellow-1 through z-yellow-5
  183-208: gray-0 tokens a-gray-0 through z-gray-0
  209-234: gray-1 tokens a-gray-1 through z-gray-1
  235-260: gray-2 tokens a-gray-2 through z-gray-2
  261-264: specials [bos] [sep] [pad] [eos]
"""

from __future__ import annotations

from mm_wordle.game import GameState, LetterFeedback

_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

_TOKEN_TO_ID: dict[str, int] = {}
_ID_TO_TOKEN: dict[int, str] = {}

_next_id = 0

for ch in _LETTERS:
    _TOKEN_TO_ID[ch] = _next_id
    _ID_TO_TOKEN[_next_id] = ch
    _next_id += 1

_TOKEN_TO_ID["?"] = _next_id
_ID_TO_TOKEN[_next_id] = "?"
_UNKNOWN_ID = _next_id
_next_id += 1

for ch in _LETTERS:
    tok = f"{ch}-green"
    _TOKEN_TO_ID[tok] = _next_id
    _ID_TO_TOKEN[_next_id] = tok
    _next_id += 1

for pos in range(1, 6):
    for ch in _LETTERS:
        tok = f"{ch}-yellow-{pos}"
        _TOKEN_TO_ID[tok] = _next_id
        _ID_TO_TOKEN[_next_id] = tok
        _next_id += 1

for count in range(3):
    for ch in _LETTERS:
        tok = f"{ch}-gray-{count}"
        _TOKEN_TO_ID[tok] = _next_id
        _ID_TO_TOKEN[_next_id] = tok
        _next_id += 1

for special in ["[bos]", "[sep]", "[pad]", "[eos]"]:
    _TOKEN_TO_ID[special] = _next_id
    _ID_TO_TOKEN[_next_id] = special
    _next_id += 1

_VOCAB_SIZE = _next_id
_LETTER_IDS = set(range(26))


class ConstraintTokenizer:
    """Tokenizer that encodes Wordle game state as a dense constraint summary."""

    @property
    def vocab_size(self) -> int:
        return _VOCAB_SIZE

    @property
    def bos_id(self) -> int:
        return _TOKEN_TO_ID["[bos]"]

    @property
    def sep_id(self) -> int:
        return _TOKEN_TO_ID["[sep]"]

    @property
    def pad_id(self) -> int:
        return _TOKEN_TO_ID["[pad]"]

    @property
    def unknown_id(self) -> int:
        return _UNKNOWN_ID

    @property
    def letter_ids(self) -> set[int]:
        return _LETTER_IDS

    def encode_token(self, token: str) -> int:
        return _TOKEN_TO_ID[token]

    def decode_token(self, token_id: int) -> str:
        return _ID_TO_TOKEN[token_id]

    def encode_game_state(self, state: GameState) -> list[int]:
        """Encode a game state as a constraint-state prompt.

        Format: [bos] <5 green slots> [sep] <sorted facts>
        """
        greens: list[str | None] = [None] * 5
        yellows: list[tuple[str, int]] = []
        gray_counts: dict[str, int] = {}

        for gf in state.guesses:
            guess = gf.guess
            feedback = gf.feedback

            green_count: dict[str, int] = {}
            yellow_count: dict[str, int] = {}

            for i, (ch, fb) in enumerate(zip(guess, feedback, strict=True)):
                if fb == LetterFeedback.GREEN:
                    greens[i] = ch
                    green_count[ch] = green_count.get(ch, 0) + 1

            for i, (ch, fb) in enumerate(zip(guess, feedback, strict=True)):
                if fb == LetterFeedback.YELLOW:
                    yellows.append((ch, i + 1))
                    yellow_count[ch] = yellow_count.get(ch, 0) + 1
                elif fb == LetterFeedback.GRAY:
                    matched = green_count.get(ch, 0) + yellow_count.get(ch, 0)
                    gray_counts[ch] = matched

        ids: list[int] = [self.bos_id]

        for i in range(5):
            if greens[i] is not None:
                ids.append(_TOKEN_TO_ID[f"{greens[i]}-green"])
            else:
                ids.append(_UNKNOWN_ID)

        ids.append(self.sep_id)

        facts: list[str] = []
        seen_yellows: set[tuple[str, int]] = set()
        for ch, pos in yellows:
            if (ch, pos) not in seen_yellows:
                facts.append(f"{ch}-yellow-{pos}")
                seen_yellows.add((ch, pos))

        for ch, count in gray_counts.items():
            count = min(count, 2)
            facts.append(f"{ch}-gray-{count}")

        facts.sort()

        for fact in facts:
            ids.append(_TOKEN_TO_ID[fact])

        return ids

    def decode_letters(self, token_ids: list[int]) -> str:
        """Decode a list of plain letter token IDs to a string."""
        return "".join(_ID_TO_TOKEN[tid] for tid in token_ids if tid in _LETTER_IDS)
