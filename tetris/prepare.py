"""Autoresearch prepare script for Tetris RL.

Validates the environment works and prints baseline metrics (random agent).
Run once before starting the autoresearch loop.
"""

import random

import mm_tetris


def random_baseline(num_games: int = 100) -> tuple[float, float]:
    """Play games with random actions to establish a baseline."""
    total_lines = 0
    total_score = 0

    for game_idx in range(num_games):
        state = mm_tetris.reset(seed=10000 + game_idx)
        steps = 0
        while not state.game_over and steps < 500:
            mask = mm_tetris.valid_action_mask(state)
            valid = [i for i, v in enumerate(mask) if v]
            if not valid:
                break
            action = random.choice(valid)
            state = mm_tetris.step(state, action)
            steps += 1
        total_lines += state.lines_cleared
        total_score += state.score

    return total_lines / num_games, total_score / num_games


def main() -> None:
    print("Tetris environment check:")
    print(f"  Grid: {mm_tetris.GRID_WIDTH}x{mm_tetris.GRID_HEIGHT}")
    print(f"  Pieces: {mm_tetris.PIECE_NAMES}")
    print(f"  Actions: {mm_tetris.NUM_ACTIONS} (4 rotations x 10 columns)")
    print(f"  State size: {mm_tetris.STATE_SIZE}")
    print()

    state = mm_tetris.reset(seed=42)
    print("Initial board:")
    print(mm_tetris.render(state))
    print()

    print("Running random baseline (100 games)...")
    avg_lines, avg_score = random_baseline()
    print(f"  Random agent: avg_lines={avg_lines:.2f}, avg_score={avg_score:.1f}")
    print()
    print("Preparation complete. Ready for autoresearch.")


if __name__ == "__main__":
    main()
