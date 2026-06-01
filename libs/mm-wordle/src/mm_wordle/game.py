"""Wordle game state and environment."""

import random
from dataclasses import dataclass, field
from enum import Enum

from mm_wordle.words import all_valid_words, load_answers


class LetterFeedback(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    GRAY = "gray"


@dataclass
class GuessFeedback:
    guess: str
    feedback: list[LetterFeedback]


@dataclass
class GameState:
    target: str
    guesses: list[GuessFeedback] = field(default_factory=list)
    turn: int = 0
    solved: bool = False
    failed: bool = False


MAX_GUESSES = 6


class WordleEnv:
    """Wordle game environment."""

    def __init__(self) -> None:
        self._valid_words: set[str] | None = None
        self._answers: list[str] | None = None

    @property
    def valid_words(self) -> set[str]:
        if self._valid_words is None:
            self._valid_words = all_valid_words()
        return self._valid_words

    @property
    def answers(self) -> list[str]:
        if self._answers is None:
            self._answers = load_answers()
        return self._answers

    def reset(self, target_word: str | None = None) -> GameState:
        """Start a new game. If target_word is None, pick a random answer."""
        if target_word is None:
            target_word = random.choice(self.answers)
        return GameState(target=target_word)

    def step(self, state: GameState, guess: str) -> tuple[GameState, bool]:
        """Submit a guess. Returns (new_state, done)."""
        guess = guess.lower()
        feedback = self.compute_feedback(guess, state.target)
        guess_feedback = GuessFeedback(guess=guess, feedback=feedback)

        new_guesses = [*state.guesses, guess_feedback]
        new_turn = state.turn + 1
        solved = guess == state.target
        failed = not solved and new_turn >= MAX_GUESSES

        new_state = GameState(
            target=state.target,
            guesses=new_guesses,
            turn=new_turn,
            solved=solved,
            failed=failed,
        )
        done = solved or failed
        return new_state, done

    @staticmethod
    def compute_feedback(guess: str, target: str) -> list[LetterFeedback]:
        """Compute letter-by-letter feedback for a guess against a target.

        Handles duplicate letters correctly:
        1. Greens are assigned first (exact position matches).
        2. Yellows are assigned left-to-right for remaining unmatched target letters.
        """
        if len(guess) != len(target):
            return [LetterFeedback.GRAY] * len(target)
        n = len(guess)
        feedback = [LetterFeedback.GRAY] * n

        # Track which target positions have been matched
        target_matched = [False] * n

        # Pass 1: assign greens
        for i in range(n):
            if guess[i] == target[i]:
                feedback[i] = LetterFeedback.GREEN
                target_matched[i] = True

        # Pass 2: assign yellows left-to-right
        for i in range(n):
            if feedback[i] == LetterFeedback.GREEN:
                continue
            for j in range(n):
                if not target_matched[j] and guess[i] == target[j]:
                    feedback[i] = LetterFeedback.YELLOW
                    target_matched[j] = True
                    break

        return feedback

    def render(self, state: GameState) -> str:
        """Render the game state as a human-readable string."""
        lines: list[str] = []
        emoji_map = {
            LetterFeedback.GREEN: "\U0001f7e9",  # green square
            LetterFeedback.YELLOW: "\U0001f7e8",  # yellow square
            LetterFeedback.GRAY: "⬜",  # white square
        }

        for gf in state.guesses:
            letters = " ".join(c.upper() for c in gf.guess)
            emojis = "".join(emoji_map[f] for f in gf.feedback)
            lines.append(f"{letters}  {emojis}")

        if state.solved:
            lines.append(f"Solved in {state.turn} guess{'es' if state.turn != 1 else ''}!")
        elif state.failed:
            lines.append(f"Failed! The word was {state.target.upper()}.")
        else:
            lines.append(f"Turn {state.turn}/{MAX_GUESSES}")

        return "\n".join(lines)
