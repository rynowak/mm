"""Load the canonical V3 hold-out split (the single source of truth).

Every phase loads the split through here so pre-train, SFT, RL, and eval agree on
the same hold-out. Regenerate the file with ``wordle3/make_split.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPLIT_PATH = Path(__file__).parent / "data" / "split.json"


@dataclass(frozen=True)
class Split:
    seed: int
    holdout_frac: float
    train_answers: list[str]
    holdout: list[str]


def load_split(path: str | Path = DEFAULT_SPLIT_PATH) -> Split:
    data = json.loads(Path(path).read_text())
    return Split(
        seed=data["seed"],
        holdout_frac=data["holdout_frac"],
        train_answers=data["train_answers"],
        holdout=data["holdout"],
    )
