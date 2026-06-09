"""mm-wordle: Wordle game environment, word lists, and reward function."""

from mm_wordle.constraint_tokenizer import ConstraintTokenizer
from mm_wordle.game import GameState, GuessFeedback, LetterFeedback, WordleEnv
from mm_wordle.golden import GoldenSolver, play_golden_game
from mm_wordle.pattern import PatternMatrix
from mm_wordle.reward import INVALID_WORD_PENALTY, compute_reward, precompute_expected_info_gain
from mm_wordle.serialize import game_state_to_prompt, game_state_to_tokens
from mm_wordle.transcripts import PretrainExample, examples_from_game, generate_examples
from mm_wordle.trie import WordTrie
from mm_wordle.words import (
    all_valid_words,
    load_answers,
    load_full_word_set,
    load_valid_guesses,
    split_answers,
)

__all__ = [
    "ConstraintTokenizer",
    "GameState",
    "GoldenSolver",
    "GuessFeedback",
    "LetterFeedback",
    "PatternMatrix",
    "WordleEnv",
    "play_golden_game",
    "all_valid_words",
    "INVALID_WORD_PENALTY",
    "compute_reward",
    "precompute_expected_info_gain",
    "game_state_to_prompt",
    "game_state_to_tokens",
    "load_answers",
    "load_full_word_set",
    "load_valid_guesses",
    "split_answers",
    "PretrainExample",
    "examples_from_game",
    "generate_examples",
    "WordTrie",
]
