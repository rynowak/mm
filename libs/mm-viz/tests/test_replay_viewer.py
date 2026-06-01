"""Tests for checkpoint replay viewer."""

from __future__ import annotations

from pathlib import Path

from mm_viz.data import EvalSnapshot, GameReplay
from mm_viz.replay_viewer import render_checkpoint_comparison, render_progress_report


def _make_replay(target: str = "crane", solved: bool = True, turns: int = 2) -> GameReplay:
    if solved:
        guesses = ["slate", target]
        feedback = [
            ["gray", "gray", "gray", "gray", "green"],
            ["green", "green", "green", "green", "green"],
        ]
    else:
        guesses = ["slate", "round", "flimp", "guess", "wrong", "nopes"]
        feedback = [["gray"] * 5] * 6
        turns = 6
    return GameReplay(
        target=target,
        guesses=guesses,
        feedback=feedback,
        solved=solved,
        turns=turns,
    )


def _make_snapshot(step: int, win_rate: float, targets: list[str] | None = None) -> EvalSnapshot:
    if targets is None:
        targets = ["crane", "house"]
    replays = [_make_replay(target=t, solved=(i % 2 == 0)) for i, t in enumerate(targets)]
    return EvalSnapshot(
        step=step,
        checkpoint_path=f"/checkpoints/step{step}",
        win_rate=win_rate,
        avg_guesses=3.5 - win_rate,
        replays=replays,
    )


class TestRenderCheckpointComparison:
    def test_creates_valid_html_file(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3, ["crane", "house"]),
            _make_snapshot(500, 0.6, ["crane", "table"]),
        ]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        assert "<html>" in content
        assert "</html>" in content
        assert "Checkpoint Comparison" in content

    def test_contains_step_info(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3),
            _make_snapshot(500, 0.6),
        ]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        assert "Step 100" in content
        assert "Step 500" in content

    def test_contains_shared_target_comparison(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3, ["crane", "house"]),
            _make_snapshot(500, 0.6, ["crane", "table"]),
        ]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        # "crane" is shared across both snapshots
        assert "CRANE" in content

    def test_contains_summary_table(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3),
            _make_snapshot(500, 0.6),
        ]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        assert "Summary" in content
        assert "Win Rate" in content

    def test_no_shared_targets(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3, ["crane"]),
            _make_snapshot(500, 0.6, ["table"]),
        ]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        assert "No shared target words" in content

    def test_single_snapshot(self, tmp_path: Path) -> None:
        snapshots = [_make_snapshot(100, 0.5)]
        output = str(tmp_path / "comparison.html")
        render_checkpoint_comparison(snapshots, output)

        content = Path(output).read_text()
        assert "<html>" in content


class TestRenderProgressReport:
    def test_creates_valid_html_file(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3),
            _make_snapshot(300, 0.5),
            _make_snapshot(500, 0.7),
        ]
        output = str(tmp_path / "progress.html")
        render_progress_report(snapshots, output)

        content = Path(output).read_text()
        assert "<html>" in content
        assert "</html>" in content
        assert "Training Progress" in content

    def test_contains_win_rate_info(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3),
            _make_snapshot(500, 0.7),
        ]
        output = str(tmp_path / "progress.html")
        render_progress_report(snapshots, output)

        content = Path(output).read_text()
        assert "Win Rate" in content
        assert "30.0%" in content
        assert "70.0%" in content

    def test_contains_avg_guesses(self, tmp_path: Path) -> None:
        snapshots = [_make_snapshot(100, 0.3)]
        output = str(tmp_path / "progress.html")
        render_progress_report(snapshots, output)

        content = Path(output).read_text()
        assert "Average Guesses" in content

    def test_sample_games_section(self, tmp_path: Path) -> None:
        snapshots = [
            _make_snapshot(100, 0.3),
            _make_snapshot(300, 0.5),
            _make_snapshot(500, 0.7),
        ]
        output = str(tmp_path / "progress.html")
        render_progress_report(snapshots, output)

        content = Path(output).read_text()
        assert "Sample Games" in content

    def test_empty_snapshots(self, tmp_path: Path) -> None:
        output = str(tmp_path / "progress.html")
        render_progress_report([], output)

        content = Path(output).read_text()
        assert "No snapshots provided" in content

    def test_sorts_by_step(self, tmp_path: Path) -> None:
        # Provide snapshots out of order
        snapshots = [
            _make_snapshot(500, 0.7),
            _make_snapshot(100, 0.3),
        ]
        output = str(tmp_path / "progress.html")
        render_progress_report(snapshots, output)

        content = Path(output).read_text()
        # Step 100 should appear before Step 500 in the output
        idx_100 = content.index("Step 100")
        idx_500 = content.index("Step 500")
        assert idx_100 < idx_500
