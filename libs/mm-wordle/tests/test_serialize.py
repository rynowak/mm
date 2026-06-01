"""Tests for game state serialization."""

from mm_wordle.game import GameState, GuessFeedback, LetterFeedback, WordleEnv
from mm_wordle.serialize import game_state_to_tokens

GREEN = LetterFeedback.GREEN
YELLOW = LetterFeedback.YELLOW
GRAY = LetterFeedback.GRAY


def test_empty_game():
    state = GameState(target="crane")
    tokens = game_state_to_tokens(state)
    assert tokens == ["[bos]", "[eos]"]


def test_single_guess():
    state = GameState(
        target="crane",
        guesses=[
            GuessFeedback(
                guess="house",
                feedback=[GRAY, GRAY, GRAY, GRAY, GREEN],
            )
        ],
        turn=1,
    )
    tokens = game_state_to_tokens(state)
    assert tokens == [
        "[bos]",
        "h",
        "o",
        "u",
        "s",
        "e",
        "[gray]",
        "[gray]",
        "[gray]",
        "[gray]",
        "[green]",
        "[eos]",
    ]


def test_two_guesses():
    state = GameState(
        target="crane",
        guesses=[
            GuessFeedback(
                guess="house",
                feedback=[GRAY, GRAY, GRAY, GRAY, GREEN],
            ),
            GuessFeedback(
                guess="crane",
                feedback=[GREEN, GREEN, GREEN, GREEN, GREEN],
            ),
        ],
        turn=2,
        solved=True,
    )
    tokens = game_state_to_tokens(state)
    assert tokens == [
        "[bos]",
        "h",
        "o",
        "u",
        "s",
        "e",
        "[gray]",
        "[gray]",
        "[gray]",
        "[gray]",
        "[green]",
        "[sep]",
        "c",
        "r",
        "a",
        "n",
        "e",
        "[green]",
        "[green]",
        "[green]",
        "[green]",
        "[green]",
        "[eos]",
    ]


def test_yellow_tokens():
    state = GameState(
        target="crane",
        guesses=[
            GuessFeedback(
                guess="earns",
                feedback=[YELLOW, YELLOW, YELLOW, GREEN, GRAY],
            )
        ],
        turn=1,
    )
    tokens = game_state_to_tokens(state)
    assert tokens == [
        "[bos]",
        "e",
        "a",
        "r",
        "n",
        "s",
        "[yellow]",
        "[yellow]",
        "[yellow]",
        "[green]",
        "[gray]",
        "[eos]",
    ]


def test_roundtrip_with_env():
    """Integration test: play a game and serialize."""
    env = WordleEnv()
    state = env.reset(target_word="crane")
    state, _ = env.step(state, "house")
    state, _ = env.step(state, "crane")

    tokens = game_state_to_tokens(state)
    assert tokens[0] == "[bos]"
    assert tokens[-1] == "[eos]"
    # Two guesses with separator: [bos] + 10 + [sep] + 10 + [eos] = 23
    assert len(tokens) == 23
    assert "[sep]" in tokens
