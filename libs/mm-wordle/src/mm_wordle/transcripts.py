"""Generate Wordle game transcripts for pre-training data."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from mm_wordle.game import WordleEnv
from mm_wordle.serialize import game_state_to_tokens
from mm_wordle.solver import play_game_decent, play_game_good, play_game_random
from mm_wordle.words import load_answers, load_valid_guesses

if TYPE_CHECKING:
    from mm_tokenizers import CharTokenizer


def generate_transcripts(
    tokenizer: CharTokenizer,
    n_games: int = 5000,
    mix: tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> list[int]:
    """Generate tokenized Wordle game transcripts at mixed skill levels.

    Args:
        tokenizer: The character tokenizer.
        n_games: Total number of games to generate.
        mix: (random_fraction, decent_fraction, good_fraction).

    Returns:
        Flat list of token IDs (games separated by [bos]...[eos]).
    """
    answers = load_answers()
    valid_guesses = list(load_valid_guesses()) + list(answers)
    env = WordleEnv()

    n_random = int(n_games * mix[0])
    n_decent = int(n_games * mix[1])
    tokens: list[int] = []
    targets = [random.choice(answers) for _ in range(n_games)]

    for i, target in enumerate(targets):
        if i < n_random:
            state = play_game_random(env, target, valid_guesses)
        elif i < n_random + n_decent:
            state = play_game_decent(env, target, valid_guesses)
        else:
            state = play_game_good(env, target, answers, valid_guesses)

        transcript_tokens = game_state_to_tokens(state)
        encoded = tokenizer.encode("".join(transcript_tokens))
        tokens.extend(encoded)

    return tokens
