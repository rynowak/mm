"""Game state serialization to token sequences."""

from mm_wordle.game import GameState, LetterFeedback

FEEDBACK_TOKEN_MAP = {
    LetterFeedback.GREEN: "[green]",
    LetterFeedback.YELLOW: "[yellow]",
    LetterFeedback.GRAY: "[gray]",
}


def game_state_to_tokens(state: GameState) -> list[str]:
    """Serialize a completed game as a full transcript for pre-training.

    Format: [bos] guess+feedback [sep] guess+feedback ... guess+feedback
    [bos] delimits game boundaries. No [eos] needed.
    """
    tokens: list[str] = ["[bos]"]

    for i, gf in enumerate(state.guesses):
        if i > 0:
            tokens.append("[sep]")

        for c in gf.guess:
            tokens.append(c)

        for fb in gf.feedback:
            tokens.append(FEEDBACK_TOKEN_MAP[fb])

    return tokens


def game_state_to_prompt(state: GameState) -> list[str]:
    """Serialize a game state as a prompt for the model to continue.

    Turn 1 (no guesses): [bos]
    Turn 2+: [bos] guess+feedback [sep] ... guess+feedback [sep]

    Ends with [sep] (after turn 1) so the model knows to produce a letter next.
    """
    tokens: list[str] = ["[bos]"]

    for i, gf in enumerate(state.guesses):
        if i > 0:
            tokens.append("[sep]")

        for c in gf.guess:
            tokens.append(c)

        for fb in gf.feedback:
            tokens.append(FEEDBACK_TOKEN_MAP[fb])

    if state.guesses:
        tokens.append("[sep]")

    return tokens
