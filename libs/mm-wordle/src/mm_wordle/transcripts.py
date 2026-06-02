"""Generate Wordle pre-training examples from game transcripts."""

from __future__ import annotations

import random
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
    """Extract per-turn pre-training examples from a completed game.

    Each turn produces one example:
      prompt = game state up to that turn (what the model sees)
      target = the 5 letter token IDs of the next guess (what the model predicts)
    """
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


def generate_examples(
    tokenizer: CharTokenizer,
    n_games: int = 20000,
    mix: tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> list[PretrainExample]:
    """Generate pre-training examples from games at mixed skill levels.

    Each game produces one example per turn. A 4-turn game produces 4 examples.
    """
    answers = load_answers()
    valid_guesses = list(load_valid_guesses()) + list(answers)
    env = WordleEnv()

    n_random = int(n_games * mix[0])
    n_decent = int(n_games * mix[1])

    examples: list[PretrainExample] = []
    targets = [random.choice(answers) for _ in range(n_games)]

    for i, target in enumerate(targets):
        if i < n_random:
            state = play_game_random(env, target, valid_guesses)
        elif i < n_random + n_decent:
            state = play_game_decent(env, target, valid_guesses)
        else:
            state = play_game_good(env, target, answers, valid_guesses)

        examples.extend(examples_from_game(state, tokenizer))

    return examples
