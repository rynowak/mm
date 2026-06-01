"""JSON-serializable data models for training visualization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class CompletionData:
    """A single completion from a GRPO group."""

    tokens: list[str]
    text: str
    log_probs: list[float]
    reward: float
    reward_breakdown: dict[str, float]


@dataclass
class GRPOStepData:
    """Full snapshot of one GRPO training step."""

    step: int
    game_state_tokens: list[str]
    game_state_text: str
    completions: list[CompletionData]
    rewards: list[float]
    advantages: list[float]
    group_mean: float
    group_std: float
    old_probs: list[float]
    new_probs: list[float]
    kl_divergence: float

    def save(self, path: Path) -> None:
        """Serialize to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> GRPOStepData:
        """Deserialize from JSON file, reconstructing nested CompletionData."""
        raw = json.loads(path.read_text())
        raw["completions"] = [CompletionData(**c) for c in raw["completions"]]
        return cls(**raw)


@dataclass
class GameReplay:
    """Replay of a single Wordle game."""

    target: str
    guesses: list[str]
    feedback: list[list[str]]  # e.g. [["green","gray",...], ...]
    solved: bool
    turns: int

    def save(self, path: Path) -> None:
        """Serialize to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> GameReplay:
        """Deserialize from JSON file."""
        return cls(**json.loads(path.read_text()))


@dataclass
class EvalSnapshot:
    """Evaluation results at a training checkpoint."""

    step: int
    checkpoint_path: str
    win_rate: float
    avg_guesses: float
    replays: list[GameReplay]

    def save(self, path: Path) -> None:
        """Serialize to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> EvalSnapshot:
        """Deserialize from JSON file, reconstructing nested GameReplay."""
        raw = json.loads(path.read_text())
        raw["replays"] = [GameReplay(**r) for r in raw["replays"]]
        return cls(**raw)
