"""Parity tests for the promoted constraint-state tokenizer (mirrors the V2 spec)."""

from mm_wordle.constraint_tokenizer import ConstraintTokenizer
from mm_wordle.game import GameState, WordleEnv


def _play(env: WordleEnv, target: str, guesses: list[str]) -> GameState:
    state = env.reset(target_word=target)
    for guess in guesses:
        state, _ = env.step(state, guess)
    return state


class TestConstraintTokenizer:
    def test_vocab_size(self) -> None:
        assert ConstraintTokenizer().vocab_size == 265

    def test_empty_state(self) -> None:
        tok = ConstraintTokenizer()
        state = WordleEnv().reset(target_word="crane")
        ids = tok.encode_game_state(state)
        assert len(ids) == 7
        assert ids[0] == tok.bos_id
        assert ids[1:6] == [tok.unknown_id] * 5
        assert ids[6] == tok.sep_id

    def test_empty_prompt_matches_empty_state(self) -> None:
        tok = ConstraintTokenizer()
        state = WordleEnv().reset(target_word="crane")
        assert tok.empty_prompt() == tok.encode_game_state(state)

    def test_green_positions(self) -> None:
        tok = ConstraintTokenizer()
        state = _play(WordleEnv(), "crane", ["crane"])
        ids = tok.encode_game_state(state)
        assert len(ids) == 7
        for i, ch in enumerate("crane"):
            assert tok.decode_token(ids[i + 1]) == f"{ch}-green"

    def test_yellow_with_position(self) -> None:
        tok = ConstraintTokenizer()
        state = _play(WordleEnv(), "crane", ["alert"])
        tokens = [tok.decode_token(i) for i in tok.encode_game_state(state)]
        assert "a-yellow-1" in tokens
        assert "e-yellow-3" in tokens
        assert "r-yellow-4" in tokens

    def test_repeated_letter_gray_count(self) -> None:
        tok = ConstraintTokenizer()
        state = _play(WordleEnv(), "foggy", ["ovoid"])
        tokens = [tok.decode_token(i) for i in tok.encode_game_state(state)]
        assert "o-yellow-1" in tokens
        assert "o-gray-1" in tokens
        assert "d-gray-0" in tokens
        assert "i-gray-0" in tokens
        assert "v-gray-0" in tokens

    def test_accumulated_across_turns_sorted_facts(self) -> None:
        tok = ConstraintTokenizer()
        state = _play(WordleEnv(), "foggy", ["slate", "humor"])
        tokens = [tok.decode_token(i) for i in tok.encode_game_state(state)]
        assert "s-gray-0" in tokens
        assert "o-yellow-4" in tokens
        fact_tokens = tokens[7:]
        assert fact_tokens == sorted(fact_tokens)

    def test_decode_letters(self) -> None:
        tok = ConstraintTokenizer()
        letter_ids = [tok.encode_token(ch) for ch in "crane"]
        assert tok.decode_letters(letter_ids) == "crane"

    def test_only_letters_in_letter_ids(self) -> None:
        tok = ConstraintTokenizer()
        assert len(tok.letter_ids) == 26
        for lid in tok.letter_ids:
            assert "a" <= tok.decode_token(lid) <= "z"
