"""Tetris environment with placement-based action space.

Actions encode (rotation, column) pairs. The piece drops to the lowest valid
row at the specified column and rotation. This simplifies the RL problem
compared to frame-by-frame key presses.
"""

import random
from dataclasses import dataclass

from mm_tetris.pieces import NUM_PIECE_TYPES, NUM_ROTATIONS, PIECE_NAMES, ROTATIONS, PieceShape

GRID_WIDTH: int = 10
GRID_HEIGHT: int = 20
NUM_ACTIONS: int = NUM_ROTATIONS * GRID_WIDTH  # 40


def action_to_placement(action: int) -> tuple[int, int]:
    """Decode action index to (rotation, column)."""
    return divmod(action, GRID_WIDTH)


def placement_to_action(rotation: int, column: int) -> int:
    """Encode (rotation, column) as action index."""
    return rotation * GRID_WIDTH + column


@dataclass
class TetrisState:
    grid: list[list[bool]]
    current_piece: str
    score: int
    lines_cleared: int
    pieces_placed: int
    game_over: bool
    seed: int

    def column_heights(self) -> list[int]:
        """Height of each column (0 = empty, GRID_HEIGHT = full)."""
        heights: list[int] = []
        for col in range(GRID_WIDTH):
            h = 0
            for row in range(GRID_HEIGHT):
                if self.grid[row][col]:
                    h = GRID_HEIGHT - row
                    break
            heights.append(h)
        return heights

    def count_holes(self) -> int:
        """Count empty cells with at least one filled cell above them."""
        holes = 0
        for col in range(GRID_WIDTH):
            found_block = False
            for row in range(GRID_HEIGHT):
                if self.grid[row][col]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def max_height(self) -> int:
        return max(self.column_heights())


def _empty_grid() -> list[list[bool]]:
    return [[False] * GRID_WIDTH for _ in range(GRID_HEIGHT)]


def _pick_piece(seed: int, piece_number: int) -> str:
    rng = random.Random(seed * 10007 + piece_number)
    return rng.choice(PIECE_NAMES)


def reset(seed: int = 42) -> TetrisState:
    """Create a fresh game state."""
    return TetrisState(
        grid=_empty_grid(),
        current_piece=_pick_piece(seed, 0),
        score=0,
        lines_cleared=0,
        pieces_placed=0,
        game_over=False,
        seed=seed,
    )


def valid_action_mask(state: TetrisState) -> list[bool]:
    """Return a boolean mask over all NUM_ACTIONS actions. True = valid."""
    mask = [False] * NUM_ACTIONS
    if state.game_over:
        return mask

    for action in range(NUM_ACTIONS):
        rotation, col = action_to_placement(action)
        shape = ROTATIONS[state.current_piece][rotation]
        if col + shape.width > GRID_WIDTH:
            continue
        drop_row = _find_drop_row(state.grid, shape, col)
        if drop_row < 0:
            continue
        mask[action] = True

    return mask


def _find_drop_row(grid: list[list[bool]], shape: PieceShape, col: int) -> int:
    """Drop piece from top, return the row where it lands. Returns -1 if blocked at spawn."""
    last_valid = -1
    for row in range(GRID_HEIGHT - shape.height + 1):
        if _fits(grid, shape, row, col):
            last_valid = row
        else:
            break
    return last_valid


def _fits(grid: list[list[bool]], shape: PieceShape, row: int, col: int) -> bool:
    """Check if a piece fits at (row, col) without overlap or out-of-bounds."""
    for dr, dc in shape.cells:
        r, c = row + dr, col + dc
        if r < 0 or r >= GRID_HEIGHT or c < 0 or c >= GRID_WIDTH:
            return False
        if grid[r][c]:
            return False
    return True


def step(state: TetrisState, action: int) -> TetrisState:
    """Place the current piece and advance the game. Returns new state."""
    if state.game_over:
        return state

    rotation, col = action_to_placement(action)
    shape = ROTATIONS[state.current_piece][rotation]

    if col + shape.width > GRID_WIDTH:
        return TetrisState(
            grid=[row[:] for row in state.grid],
            current_piece=state.current_piece,
            score=state.score,
            lines_cleared=state.lines_cleared,
            pieces_placed=state.pieces_placed,
            game_over=True,
            seed=state.seed,
        )

    drop_row = _find_drop_row(state.grid, shape, col)
    if drop_row < 0:
        return TetrisState(
            grid=[row[:] for row in state.grid],
            current_piece=state.current_piece,
            score=state.score,
            lines_cleared=state.lines_cleared,
            pieces_placed=state.pieces_placed,
            game_over=True,
            seed=state.seed,
        )

    new_grid = [row[:] for row in state.grid]
    for dr, dc in shape.cells:
        new_grid[drop_row + dr][col + dc] = True

    cleared = _clear_lines(new_grid)

    # Standard Tetris scoring: 0, 100, 300, 500, 800
    score_table = [0, 100, 300, 500, 800]
    added_score = score_table[cleared] if cleared < len(score_table) else cleared * 200

    next_piece_num = state.pieces_placed + 1
    next_piece = _pick_piece(state.seed, next_piece_num)

    new_state = TetrisState(
        grid=new_grid,
        current_piece=next_piece,
        score=state.score + added_score,
        lines_cleared=state.lines_cleared + cleared,
        pieces_placed=next_piece_num,
        game_over=False,
        seed=state.seed,
    )

    if not any(valid_action_mask(new_state)):
        new_state.game_over = True

    return new_state


def _clear_lines(grid: list[list[bool]]) -> int:
    """Remove full rows from the grid, shifting everything down. Returns count."""
    cleared = 0
    row = GRID_HEIGHT - 1
    while row >= 0:
        if all(grid[row]):
            del grid[row]
            grid.insert(0, [False] * GRID_WIDTH)
            cleared += 1
        else:
            row -= 1
    return cleared


def render(state: TetrisState) -> str:
    """Render the grid as a string for debugging."""
    lines: list[str] = []
    lines.append("+" + "-" * GRID_WIDTH + "+")
    for row in state.grid:
        cells = "".join("#" if c else "." for c in row)
        lines.append("|" + cells + "|")
    lines.append("+" + "-" * GRID_WIDTH + "+")
    lines.append(f"Piece: {state.current_piece}  Score: {state.score}  Lines: {state.lines_cleared}")
    return "\n".join(lines)


def state_to_flat(state: TetrisState) -> list[float]:
    """Convert state to a flat feature vector for neural network input.

    Returns: list of floats, length = GRID_HEIGHT * GRID_WIDTH + NUM_PIECE_TYPES (207)
    """
    grid_flat = [float(cell) for row in state.grid for cell in row]
    piece_onehot = [0.0] * NUM_PIECE_TYPES
    piece_onehot[PIECE_NAMES.index(state.current_piece)] = 1.0
    return grid_flat + piece_onehot


STATE_SIZE: int = GRID_HEIGHT * GRID_WIDTH + NUM_PIECE_TYPES  # 207
