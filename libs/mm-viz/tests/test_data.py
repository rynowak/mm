"""Tests for data model serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm_viz.data import CompletionData, EvalSnapshot, GameReplay, GRPOStepData

if TYPE_CHECKING:
    from pathlib import Path


def _make_completion(text: str = "hello") -> CompletionData:
    return CompletionData(
        tokens=list(text),
        text=text,
        log_probs=[-0.1 * (i + 1) for i in range(len(text))],
        reward=1.0,
        reward_breakdown={"format": 0.5, "correct": 0.5},
    )


def _make_replay(solved: bool = True) -> GameReplay:
    return GameReplay(
        target="crane",
        guesses=["slate", "crane"],
        feedback=[
            ["gray", "gray", "gray", "gray", "green"],
            ["green", "green", "green", "green", "green"],
        ],
        solved=solved,
        turns=2,
    )


class TestCompletionData:
    def test_create(self) -> None:
        c = _make_completion()
        assert c.text == "hello"
        assert len(c.tokens) == 5
        assert c.reward == 1.0


class TestGameReplay:
    def test_create(self) -> None:
        r = _make_replay()
        assert r.target == "crane"
        assert r.solved is True

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = _make_replay()
        path = tmp_path / "replay.json"
        original.save(path)
        loaded = GameReplay.load(path)
        assert loaded == original

    def test_roundtrip_unsolved(self, tmp_path: Path) -> None:
        original = _make_replay(solved=False)
        path = tmp_path / "replay.json"
        original.save(path)
        loaded = GameReplay.load(path)
        assert loaded == original
        assert loaded.solved is False


class TestGRPOStepData:
    def test_create(self) -> None:
        step = GRPOStepData(
            step=42,
            game_state_tokens=["a", "b"],
            game_state_text="ab",
            completions=[_make_completion()],
            rewards=[1.0],
            advantages=[0.5],
            group_mean=0.8,
            group_std=0.1,
            old_probs=[0.3],
            new_probs=[0.4],
            kl_divergence=0.01,
        )
        assert step.step == 42
        assert len(step.completions) == 1

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = GRPOStepData(
            step=10,
            game_state_tokens=["x"],
            game_state_text="x",
            completions=[_make_completion("ab"), _make_completion("cd")],
            rewards=[1.0, 0.5],
            advantages=[0.3, -0.3],
            group_mean=0.75,
            group_std=0.25,
            old_probs=[0.2, 0.3],
            new_probs=[0.25, 0.35],
            kl_divergence=0.02,
        )
        path = tmp_path / "grpo.json"
        original.save(path)
        loaded = GRPOStepData.load(path)
        assert loaded == original
        assert isinstance(loaded.completions[0], CompletionData)


class TestEvalSnapshot:
    def test_create(self) -> None:
        snap = EvalSnapshot(
            step=100,
            checkpoint_path="/checkpoints/step100",
            win_rate=0.85,
            avg_guesses=3.2,
            replays=[_make_replay()],
        )
        assert snap.step == 100
        assert len(snap.replays) == 1

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        original = EvalSnapshot(
            step=200,
            checkpoint_path="/checkpoints/step200",
            win_rate=0.9,
            avg_guesses=2.8,
            replays=[_make_replay(True), _make_replay(False)],
        )
        path = tmp_path / "eval.json"
        original.save(path)
        loaded = EvalSnapshot.load(path)
        assert loaded == original
        assert isinstance(loaded.replays[0], GameReplay)
        assert loaded.replays[0].solved is True
        assert loaded.replays[1].solved is False
