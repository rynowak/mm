"""Game state serialization to token sequences."""

from mm_wordle.game import GameState, LetterFeedback

FEEDBACK_TOKEN_MAP = {
    LetterFeedback.GREEN: "[green]",
    LetterFeedback.YELLOW: "[yellow]",
    LetterFeedback.GRAY: "[gray]",
}


def game_state_to_tokens(state: GameState) -> list[str]:
    """Serialize a game state to a list of tokens.

    Format: [bos] g u e s s [green] [gray] [yellow] [gray] [green] [sep] ... [eos]
    Each guess: 5 letter tokens then 5 feedback tokens, separated by [sep].
    """
    tokens: list[str] = ["[bos]"]

    for i, gf in enumerate(state.guesses):
        if i > 0:
            tokens.append("[sep]")

        # Add letter tokens
        for c in gf.guess:
            tokens.append(c)

        # Add feedback tokens
        for fb in gf.feedback:
            tokens.append(FEEDBACK_TOKEN_MAP[fb])

    tokens.append("[eos]")
    return tokens
