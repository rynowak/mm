"""mm-wordle: Wordle game environment, word lists, and reward function."""

from mm_wordle.game import GameState, GuessFeedback, LetterFeedback, WordleEnv
from mm_wordle.reward import compute_reward, precompute_expected_info_gain
from mm_wordle.serialize import game_state_to_prompt, game_state_to_tokens
from mm_wordle.transcripts import PretrainExample, examples_from_game, generate_examples
from mm_wordle.trie import WordTrie
from mm_wordle.words import all_valid_words, load_answers, load_valid_guesses

__all__ = [
    "GameState",
    "GuessFeedback",
    "LetterFeedback",
    "WordleEnv",
    "all_valid_words",
    "compute_reward",
    "precompute_expected_info_gain",
    "game_state_to_prompt",
    "game_state_to_tokens",
    "load_answers",
    "load_valid_guesses",
    "PretrainExample",
    "examples_from_game",
    "generate_examples",
    "WordTrie",
]
