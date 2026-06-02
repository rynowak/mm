"""Reward function for Wordle RL training."""

from dataclasses import dataclass

from mm_wordle.game import GameState, GuessFeedback, LetterFeedback
from mm_wordle.solver import filter_candidates


@dataclass
class RewardConfig:
    invalid_word: float = -1.0
    repeated_guess: float = -0.5
    contradicts_clues: float = -0.3
    no_new_info: float = 0.0
    green_letter: float = 0.2
    yellow_letter: float = 0.1
    elimination_weight: float = 1.0
    solved: float = 2.0
    failed: float = -0.5


def compute_reward(
    state: GameState,
    guess: str,
    feedback: list[LetterFeedback],
    valid_words: set[str],
    config: RewardConfig | None = None,
    candidates_before: list[str] | None = None,
) -> float:
    """Compute reward for a guess given the resulting feedback and game state.

    Args:
        state: The game state *after* the guess has been applied.
        guess: The guess that was made.
        feedback: The feedback for this guess.
        valid_words: Set of all valid words.
        config: Reward configuration. Uses defaults if None.
        candidates_before: List of candidate words before this guess.
            If provided, an elimination reward is computed based on how
            many candidates this guess rules out.

    Returns:
        The reward value.
    """
    if config is None:
        config = RewardConfig()

    guess = guess.lower()

    if guess not in valid_words:
        return config.invalid_word

    previous_guesses = [gf.guess for gf in state.guesses[:-1]] if state.guesses else []
    if guess in previous_guesses:
        return config.repeated_guess

    if state.solved:
        return config.solved

    if state.failed:
        return config.failed

    if _contradicts_clues(guess, state.guesses[:-1] if state.guesses else []):
        return config.contradicts_clues

    reward = 0.0

    # Green/yellow letter bonuses
    for fb in feedback:
        if fb == LetterFeedback.GREEN:
            reward += config.green_letter
        elif fb == LetterFeedback.YELLOW:
            reward += config.yellow_letter

    # Information-theoretic elimination reward
    if candidates_before is not None and len(candidates_before) > 1:
        candidates_after = filter_candidates(candidates_before, guess, feedback)
        n_before = len(candidates_before)
        n_after = max(len(candidates_after), 1)
        elimination_rate = 1.0 - (n_after / n_before)
        reward += config.elimination_weight * elimination_rate

    if reward == 0.0:
        return config.no_new_info

    return reward


def _contradicts_clues(
    guess: str,
    previous_guesses: list[GuessFeedback],
) -> bool:
    """Check if a guess contradicts information from previous guesses.

    A guess contradicts clues if:
    - It places a letter in a position that was previously green with a different letter.
    - It uses a letter in a position that was previously yellow for that letter.
    - It includes a letter that was previously gray (with no other occurrences).
    """
    if not previous_guesses:
        return False

    # Build knowledge from previous guesses
    known_green: dict[int, str] = {}  # position -> letter
    known_yellow: dict[int, set[str]] = {}  # position -> letters that are yellow there
    known_gray: set[str] = set()  # letters known to not be in the word
    known_present: set[str] = set()  # letters known to be in the word

    for gf in previous_guesses:
        for i, fb in enumerate(gf.feedback):
            letter = gf.guess[i]
            if fb == LetterFeedback.GREEN:
                known_green[i] = letter
                known_present.add(letter)
            elif fb == LetterFeedback.YELLOW:
                if i not in known_yellow:
                    known_yellow[i] = set()
                known_yellow[i].add(letter)
                known_present.add(letter)
            elif fb == LetterFeedback.GRAY:
                # Only mark as gray if this letter has no green/yellow elsewhere
                if letter not in known_present:
                    known_gray.add(letter)

    # Check contradictions
    for i, letter in enumerate(guess):
        # Using a known-green position with wrong letter
        if i in known_green and known_green[i] != letter:
            return True

        # Placing a letter in a position where it was yellow
        if i in known_yellow and letter in known_yellow[i]:
            return True

    # Using a letter known to be absent (gray and never green/yellow)
    return any(letter in known_gray and letter not in known_present for letter in guess)
