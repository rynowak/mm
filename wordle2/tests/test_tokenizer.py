"""Tests for V2 constraint-state tokenizer."""

from mm_wordle.game import GameState, WordleEnv

from wordle2.tokenizer import ConstraintTokenizer


def _play(env: WordleEnv, target: str, guesses: list[str]) -> GameState:
    state = env.reset(target_word=target)
    for guess in guesses:
        state, _ = env.step(state, guess)
    return state


class TestConstraintTokenizer:
    def test_vocab_size(self) -> None:
        tok = ConstraintTokenizer()
        assert tok.vocab_size == 265

    def test_empty_state(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        state = env.reset(target_word="crane")
        ids = tok.encode_game_state(state)
        # [bos] ? ? ? ? ? [sep]
        assert len(ids) == 7
        assert ids[0] == tok.bos_id
        assert ids[1:6] == [tok.unknown_id] * 5
        assert ids[6] == tok.sep_id

    def test_all_gray(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        state = _play(env, "foggy", ["slate"])
        ids = tok.encode_game_state(state)
        # [bos] ? ? ? ? ? [sep] a-gray-0 e-gray-0 l-gray-0 s-gray-0 t-gray-0
        assert ids[0] == tok.bos_id
        assert ids[1:6] == [tok.unknown_id] * 5
        assert ids[6] == tok.sep_id
        # 5 gray facts, sorted alphabetically
        assert len(ids) == 12

    def test_green_positions(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        state = _play(env, "crane", ["crane"])
        ids = tok.encode_game_state(state)
        # [bos] c-green r-green a-green n-green e-green [sep]
        assert len(ids) == 7
        for i, ch in enumerate("crane"):
            assert tok.decode_token(ids[i + 1]) == f"{ch}-green"

    def test_yellow_with_position(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        # grail vs foggy: g=gray r=gray a=yellow(pos3) i=gray l=gray
        # Wait, a is at pos 3 in grail, foggy has no a → gray
        # Let's use a better example
        state = _play(env, "crane", ["alert"])
        ids = tok.encode_game_state(state)
        tokens = [tok.decode_token(i) for i in ids]
        # alert vs crane: a=yellow(pos1) l=gray e=yellow(pos3) r=yellow(pos4) t=gray
        assert "a-yellow-1" in tokens
        assert "e-yellow-3" in tokens
        assert "r-yellow-4" in tokens

    def test_repeated_letter_gray_count(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        # ovoid vs foggy: o=yellow(pos1) v=gray o=gray(count=1) i=gray d=gray
        state = _play(env, "foggy", ["ovoid"])
        ids = tok.encode_game_state(state)
        tokens = [tok.decode_token(i) for i in ids]
        assert "o-yellow-1" in tokens
        assert "o-gray-1" in tokens  # exactly 1 'o'
        assert "d-gray-0" in tokens
        assert "i-gray-0" in tokens
        assert "v-gray-0" in tokens

    def test_accumulated_across_turns(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        state = _play(env, "foggy", ["slate", "humor"])
        ids = tok.encode_game_state(state)
        tokens = [tok.decode_token(i) for i in ids]
        # slate: all gray → a,e,l,s,t gray-0
        # humor: o=yellow(pos4), rest gray → h,m,r,u gray-0
        assert "s-gray-0" in tokens
        assert "o-yellow-4" in tokens
        assert "h-gray-0" in tokens
        # facts should be sorted
        fact_tokens = tokens[7:]  # after [bos] ????? [sep]
        assert fact_tokens == sorted(fact_tokens)

    def test_green_accumulates(self) -> None:
        tok = ConstraintTokenizer()
        env = WordleEnv()
        state = _play(env, "crane", ["crime", "crate"])
        ids = tok.encode_game_state(state)
        tokens = [tok.decode_token(i) for i in ids]
        # After crime: c=green, r=green, i=gray, m=gray, e=yellow(pos5)
        # After crate: c=green, r=green, a=green, t=gray, e=green
        assert tokens[1] == "c-green"
        assert tokens[2] == "r-green"
        assert tokens[3] == "a-green"
        assert tokens[5] == "e-green"

    def test_decode_letters(self) -> None:
        tok = ConstraintTokenizer()
        letter_ids = [tok.encode_token(ch) for ch in "crane"]
        assert tok.decode_letters(letter_ids) == "crane"

    def test_only_letters_in_letter_ids(self) -> None:
        tok = ConstraintTokenizer()
        assert len(tok.letter_ids) == 26
        for lid in tok.letter_ids:
            token = tok.decode_token(lid)
            assert len(token) == 1
            assert "a" <= token <= "z"
