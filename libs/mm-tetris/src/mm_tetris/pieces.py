"""Tetromino piece definitions with rotation states."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PieceShape:
    """A tetromino in a specific rotation, defined by cell offsets from top-left."""

    cells: tuple[tuple[int, int], ...]
    width: int
    height: int


def _shape(cells: list[tuple[int, int]]) -> PieceShape:
    width = max(c for _, c in cells) + 1
    height = max(r for r, _ in cells) + 1
    return PieceShape(cells=tuple(cells), width=width, height=height)


PIECE_NAMES: list[str] = ["I", "O", "T", "S", "Z", "J", "L"]
NUM_PIECE_TYPES: int = len(PIECE_NAMES)

# Each piece has 4 rotation states. Cells are (row, col) offsets from the
# top-left corner of the bounding box.
ROTATIONS: dict[str, list[PieceShape]] = {
    "I": [
        _shape([(0, 0), (0, 1), (0, 2), (0, 3)]),
        _shape([(0, 0), (1, 0), (2, 0), (3, 0)]),
        _shape([(0, 0), (0, 1), (0, 2), (0, 3)]),
        _shape([(0, 0), (1, 0), (2, 0), (3, 0)]),
    ],
    "O": [
        _shape([(0, 0), (0, 1), (1, 0), (1, 1)]),
        _shape([(0, 0), (0, 1), (1, 0), (1, 1)]),
        _shape([(0, 0), (0, 1), (1, 0), (1, 1)]),
        _shape([(0, 0), (0, 1), (1, 0), (1, 1)]),
    ],
    "T": [
        _shape([(0, 1), (1, 0), (1, 1), (1, 2)]),
        _shape([(0, 0), (1, 0), (1, 1), (2, 0)]),
        _shape([(0, 0), (0, 1), (0, 2), (1, 1)]),
        _shape([(0, 1), (1, 0), (1, 1), (2, 1)]),
    ],
    "S": [
        _shape([(0, 1), (0, 2), (1, 0), (1, 1)]),
        _shape([(0, 0), (1, 0), (1, 1), (2, 1)]),
        _shape([(0, 1), (0, 2), (1, 0), (1, 1)]),
        _shape([(0, 0), (1, 0), (1, 1), (2, 1)]),
    ],
    "Z": [
        _shape([(0, 0), (0, 1), (1, 1), (1, 2)]),
        _shape([(0, 1), (1, 0), (1, 1), (2, 0)]),
        _shape([(0, 0), (0, 1), (1, 1), (1, 2)]),
        _shape([(0, 1), (1, 0), (1, 1), (2, 0)]),
    ],
    "J": [
        _shape([(0, 0), (1, 0), (1, 1), (1, 2)]),
        _shape([(0, 0), (0, 1), (1, 0), (2, 0)]),
        _shape([(0, 0), (0, 1), (0, 2), (1, 2)]),
        _shape([(0, 1), (1, 1), (2, 0), (2, 1)]),
    ],
    "L": [
        _shape([(0, 2), (1, 0), (1, 1), (1, 2)]),
        _shape([(0, 0), (1, 0), (2, 0), (2, 1)]),
        _shape([(0, 0), (0, 1), (0, 2), (1, 0)]),
        _shape([(0, 0), (0, 1), (1, 1), (2, 1)]),
    ],
}

NUM_ROTATIONS: int = 4
