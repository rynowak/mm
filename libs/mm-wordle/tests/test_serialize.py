"""Tests for game state serialization."""

from mm_wordle.game import GameState, GuessFeedback, LetterFeedback, WordleEnv
from mm_wordle.serialize import game_state_to_prompt, game_state_to_tokens

GREEN = LetterFeedback.GREEN
YELLOW = LetterFeedback.YELLOW
GRAY = LetterFeedback.GRAY


def test_empty_game_transcript():
    state = GameState(target="crane")
    tokens = game_state_to_tokens(state)
    assert tokens == ["[bos]"]


def test_single_guess_transcript():
    state = GameState(
        target="crane",
        guesses=[GuessFeedback(guess="house", feedback=[GRAY, GRAY, GRAY, GRAY, GREEN])],
        turn=1,
    )
    tokens = game_state_to_tokens(state)
    assert tokens == ["[bos]", "h", "o", "u", "s", "e", "[gray]", "[gray]", "[gray]", "[gray]", "[green]"]


def test_two_guesses_transcript():
    state = GameState(
        target="crane",
        guesses=[
            GuessFeedback(guess="house", feedback=[GRAY, GRAY, GRAY, GRAY, GREEN]),
            GuessFeedback(guess="crane", feedback=[GREEN, GREEN, GREEN, GREEN, GREEN]),
        ],
        turn=2,
        solved=True,
    )
    tokens = game_state_to_tokens(state)
    assert tokens[0] == "[bos]"
    assert "[sep]" in tokens
    assert "[eos]" not in tokens
    assert len(tokens) == 22  # [bos] + 10 + [sep] + 10


def test_empty_game_prompt():
    state = GameState(target="crane")
    tokens = game_state_to_prompt(state)
    assert tokens == ["[bos]"]


def test_prompt_after_one_guess():
    state = GameState(
        target="crane",
        guesses=[GuessFeedback(guess="house", feedback=[GRAY, GRAY, GRAY, GRAY, GREEN])],
        turn=1,
    )
    tokens = game_state_to_prompt(state)
    assert tokens[0] == "[bos]"
    assert tokens[-1] == "[sep]"
    assert "[eos]" not in tokens


def test_prompt_matches_transcript_prefix():
    state = GameState(
        target="crane",
        guesses=[
            GuessFeedback(guess="house", feedback=[GRAY, GRAY, GRAY, GRAY, GREEN]),
            GuessFeedback(guess="crane", feedback=[GREEN, GREEN, GREEN, GREEN, GREEN]),
        ],
        turn=2,
        solved=True,
    )
    transcript = game_state_to_tokens(state)

    # Prompt after first guess should be a prefix of the full transcript
    state_1 = GameState(
        target="crane",
        guesses=[GuessFeedback(guess="house", feedback=[GRAY, GRAY, GRAY, GRAY, GREEN])],
        turn=1,
    )
    prompt = game_state_to_prompt(state_1)
    # prompt ends with [sep], transcript has the content after [sep]
    assert transcript[: len(prompt) - 1] == prompt[:-1]


def test_roundtrip_with_env():
    env = WordleEnv()
    state = env.reset(target_word="crane")
    state, _ = env.step(state, "house")
    state, _ = env.step(state, "crane")

    tokens = game_state_to_tokens(state)
    assert tokens[0] == "[bos]"
    assert "[eos]" not in tokens
    assert "[sep]" in tokens
