"""Simple Wordle solvers for generating training transcripts."""

from __future__ import annotations

import random

from mm_wordle.game import GameState, LetterFeedback, WordleEnv


def filter_candidates(candidates: list[str], guess: str, feedback: list[LetterFeedback]) -> list[str]:
    """Filter candidate words to only those consistent with the feedback."""
    if len(guess) != len(feedback):
        return list(candidates)
    result = []
    for word in candidates:
        if _is_consistent(word, guess, feedback):
            result.append(word)
    return result


def _is_consistent(candidate: str, guess: str, feedback: list[LetterFeedback]) -> bool:
    """Check if a candidate word is consistent with the feedback from a guess."""
    for i, (g_ch, fb) in enumerate(zip(guess, feedback, strict=True)):
        if fb == LetterFeedback.GREEN:
            if candidate[i] != g_ch:
                return False
        elif fb == LetterFeedback.YELLOW:
            if candidate[i] == g_ch:
                return False
            if g_ch not in candidate:
                return False
        elif fb == LetterFeedback.GRAY:
            count_in_guess = sum(
                1
                for j in range(len(guess))
                if guess[j] == g_ch and feedback[j] in (LetterFeedback.GREEN, LetterFeedback.YELLOW)
            )
            count_in_candidate = candidate.count(g_ch)
            if count_in_candidate > count_in_guess:
                return False
    return True


def entropy_guess(candidates: list[str], valid_guesses: list[str]) -> str:
    """Pick the guess that maximizes expected information gain.

    For each possible guess, simulate the feedback against every candidate
    and count how many distinct feedback patterns result. The guess that
    produces the most distinct patterns eliminates the most candidates on
    average.
    """
    if len(candidates) <= 2:
        return candidates[0]

    best_guess = candidates[0]
    best_score = 0

    search_space = candidates if len(candidates) <= 200 else random.sample(candidates, 200)
    for guess in search_space:
        patterns: set[tuple[str, ...]] = set()
        for target in candidates:
            fb = WordleEnv.compute_feedback(guess, target)
            patterns.add(tuple(f.value for f in fb))
        if len(patterns) > best_score:
            best_score = len(patterns)
            best_guess = guess

    return best_guess


def play_game_random(env: WordleEnv, target: str, valid_guesses: list[str]) -> GameState:
    """Play a game with random guesses."""
    state = env.reset(target_word=target)
    while not state.solved and not state.failed:
        guess = random.choice(valid_guesses)
        state, _ = env.step(state, guess)
    return state


def play_game_decent(env: WordleEnv, target: str, valid_guesses: list[str]) -> GameState:
    """Play a game that eliminates impossible words but doesn't optimize."""
    state = env.reset(target_word=target)
    candidates = list(valid_guesses)
    while not state.solved and not state.failed:
        guess = random.choice(candidates)
        state, _ = env.step(state, guess)
        fb = state.guesses[-1].feedback
        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            candidates = list(valid_guesses)
    return state


def play_game_good(env: WordleEnv, target: str, answers: list[str], valid_guesses: list[str]) -> GameState:
    """Play a game with entropy-based guess selection."""
    state = env.reset(target_word=target)
    candidates = list(answers)
    while not state.solved and not state.failed:
        guess = random.choice(candidates) if len(candidates) > 500 else entropy_guess(candidates, valid_guesses)
        state, _ = env.step(state, guess)
        fb = state.guesses[-1].feedback
        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            candidates = list(answers)
    return state
