"""Generate Wordle pre-training examples from game transcripts."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mm_wordle.game import GameState, WordleEnv
from mm_wordle.serialize import game_state_to_prompt
from mm_wordle.solver import play_game_decent, play_game_good, play_game_random
from mm_wordle.words import load_answers, load_valid_guesses

if TYPE_CHECKING:
    from mm_tokenizers import CharTokenizer


@dataclass
class PretrainExample:
    """A single pre-training example: prompt + target."""

    prompt_ids: list[int]
    target_ids: list[int]


def examples_from_game(state: GameState, tokenizer: CharTokenizer) -> list[PretrainExample]:
    """Extract per-turn pre-training examples from a completed game."""
    examples: list[PretrainExample] = []
    env = WordleEnv()
    replay_state = env.reset(target_word=state.target)

    for gf in state.guesses:
        prompt_tokens = game_state_to_prompt(replay_state)
        prompt_ids = tokenizer.encode("".join(prompt_tokens))
        target_ids = tokenizer.encode(gf.guess)

        examples.append(PretrainExample(prompt_ids=prompt_ids, target_ids=target_ids))

        replay_state, _ = env.step(replay_state, gf.guess)

    return examples


def _play_game(args: tuple[str, str, list[str], list[str]]) -> GameState:
    """Play a single game. Picklable for multiprocessing."""
    target, strategy, answers, valid_guesses = args
    env = WordleEnv()
    if strategy == "random":
        return play_game_random(env, target, valid_guesses)
    elif strategy == "decent":
        return play_game_decent(env, target, valid_guesses)
    else:
        return play_game_good(env, target, answers, valid_guesses)


def generate_examples(
    tokenizer: CharTokenizer,
    n_games: int = 20000,
    mix: tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> list[PretrainExample]:
    """Generate pre-training examples from games at mixed skill levels.

    Uses multiprocessing for parallel game generation.
    """
    answers = load_answers()
    valid_guesses = list(load_valid_guesses()) + list(answers)

    n_random = int(n_games * mix[0])
    n_decent = int(n_games * mix[1])

    targets = [random.choice(answers) for _ in range(n_games)]

    tasks: list[tuple[str, str, list[str], list[str]]] = []
    for i, target in enumerate(targets):
        if i < n_random:
            strategy = "random"
        elif i < n_random + n_decent:
            strategy = "decent"
        else:
            strategy = "good"
        tasks.append((target, strategy, list(answers), valid_guesses))

    with ProcessPoolExecutor() as pool:
        states = list(pool.map(_play_game, tasks, chunksize=100))

    examples: list[PretrainExample] = []
    for state in states:
        examples.extend(examples_from_game(state, tokenizer))

    return examples
