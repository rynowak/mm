"""Tests for strategy evolution analysis and visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mm_viz.data import EvalSnapshot, GameReplay
from mm_viz.strategy import analyze_strategy, render_strategy_html

if TYPE_CHECKING:
    from pathlib import Path


def _make_replay(
    target: str = "crane",
    first_guess: str = "slate",
    solved: bool = True,
    turns: int = 3,
) -> GameReplay:
    guesses = [first_guess]
    feedback = [["gray"] * len(target)]
    if solved:
        guesses.append(target)
        feedback.append(["green"] * len(target))
    return GameReplay(
        target=target,
        guesses=guesses,
        feedback=feedback,
        solved=solved,
        turns=turns,
    )


def _make_snapshots() -> list[EvalSnapshot]:
    return [
        EvalSnapshot(
            step=100,
            checkpoint_path="/ckpt/100",
            win_rate=0.4,
            avg_guesses=4.5,
            replays=[
                _make_replay(first_guess="slate", solved=True, turns=4),
                _make_replay(first_guess="slate", solved=True, turns=5),
                _make_replay(first_guess="adieu", solved=False, turns=6),
                _make_replay(first_guess="crane", solved=False, turns=6),
                _make_replay(first_guess="slate", solved=False, turns=6),
            ],
        ),
        EvalSnapshot(
            step=500,
            checkpoint_path="/ckpt/500",
            win_rate=0.7,
            avg_guesses=3.2,
            replays=[
                _make_replay(first_guess="crane", solved=True, turns=2),
                _make_replay(first_guess="crane", solved=True, turns=3),
                _make_replay(first_guess="crane", solved=True, turns=3),
                _make_replay(first_guess="slate", solved=True, turns=4),
                _make_replay(first_guess="adieu", solved=False, turns=6),
            ],
        ),
        EvalSnapshot(
            step=1000,
            checkpoint_path="/ckpt/1000",
            win_rate=0.9,
            avg_guesses=2.8,
            replays=[
                _make_replay(first_guess="crane", solved=True, turns=2),
                _make_replay(first_guess="crane", solved=True, turns=2),
                _make_replay(first_guess="crane", solved=True, turns=3),
                _make_replay(first_guess="crane", solved=True, turns=4),
                _make_replay(first_guess="slate", solved=False, turns=6),
            ],
        ),
    ]


# ------------------------------------------------------------------
# analyze_strategy
# ------------------------------------------------------------------


class TestAnalyzeStrategy:
    def test_returns_correct_keys(self) -> None:
        result = analyze_strategy(_make_snapshots())
        expected_keys = {
            "steps",
            "win_rates",
            "avg_guesses",
            "first_guess_diversity",
            "first_guess_distribution",
            "letter_frequency",
            "guess_distribution",
        }
        assert set(result.keys()) == expected_keys

    def test_win_rates_match_snapshot_data(self) -> None:
        snapshots = _make_snapshots()
        result = analyze_strategy(snapshots)
        assert result["win_rates"] == [s.win_rate for s in snapshots]
        assert result["steps"] == [s.step for s in snapshots]

    def test_first_guess_distribution_populated(self) -> None:
        result = analyze_strategy(_make_snapshots())
        dist = result["first_guess_distribution"]

        # Step 100: 3 slate, 1 adieu, 1 crane
        assert dist[100]["slate"] == 3
        assert dist[100]["adieu"] == 1
        assert dist[100]["crane"] == 1

        # Step 1000: 4 crane, 1 slate
        assert dist[1000]["crane"] == 4
        assert dist[1000]["slate"] == 1

    def test_guess_distribution_populated(self) -> None:
        result = analyze_strategy(_make_snapshots())
        dist = result["guess_distribution"]
        # Step 100: 2 wins (turns 4 and 5), 3 losses
        assert dist[100][4] == 1
        assert dist[100][5] == 1
        assert dist[100]["X"] == 3

    def test_first_guess_diversity(self) -> None:
        result = analyze_strategy(_make_snapshots())
        # Step 100: slate, adieu, crane = 3 unique
        assert result["first_guess_diversity"][0] == 3
        # Step 1000: crane, slate = 2 unique
        assert result["first_guess_diversity"][2] == 2

    def test_empty_snapshots(self) -> None:
        result = analyze_strategy([])
        assert result["steps"] == []
        assert result["win_rates"] == []


# ------------------------------------------------------------------
# render_strategy_html
# ------------------------------------------------------------------


class TestRenderStrategyHtml:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        out = str(tmp_path / "strategy.html")
        render_strategy_html(_make_snapshots(), out)
        with open(out) as f:
            content = f.read()
        assert "<html>" in content
        assert "</html>" in content

    def test_contains_svg_elements(self, tmp_path: Path) -> None:
        out = str(tmp_path / "strategy.html")
        render_strategy_html(_make_snapshots(), out)
        with open(out) as f:
            content = f.read()
        assert "<svg" in content
        assert "</svg>" in content
        assert "<polyline" in content

    def test_contains_win_rate_data(self, tmp_path: Path) -> None:
        out = str(tmp_path / "strategy.html")
        render_strategy_html(_make_snapshots(), out)
        with open(out) as f:
            content = f.read()
        assert "Win Rate" in content
        assert "Average Guesses" in content

    def test_contains_all_sections(self, tmp_path: Path) -> None:
        out = str(tmp_path / "strategy.html")
        render_strategy_html(_make_snapshots(), out)
        with open(out) as f:
            content = f.read()
        assert "First Guess Analysis" in content
        assert "Letter Frequency" in content
        assert "Guess Distribution" in content
