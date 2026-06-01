"""Tests for Wordle board HTML rendering."""

from __future__ import annotations

from mm_viz.board import render_comparison_html, render_game_html, render_games_report
from mm_viz.data import GameReplay


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


class TestRenderGameHtml:
    def test_contains_letters(self) -> None:
        html = render_game_html(_make_replay())
        for letter in "SLATECRANE":
            assert letter in html

    def test_contains_color_codes(self) -> None:
        html = render_game_html(_make_replay())
        assert "#6aaa64" in html  # green
        assert "#787c7e" in html  # gray

    def test_valid_html_structure(self) -> None:
        html = render_game_html(_make_replay())
        assert "<div" in html
        assert "</div>" in html


class TestRenderComparisonHtml:
    def test_multiple_boards(self) -> None:
        replays = [_make_replay(), _make_replay(solved=False)]
        labels = ["Model A", "Model B"]
        html = render_comparison_html(replays, labels)
        assert "Model A" in html
        assert "Model B" in html
        # Both boards should have letters
        assert html.count("CRANE"[0]) >= 2  # C appears in both boards

    def test_contains_all_labels(self) -> None:
        replays = [_make_replay() for _ in range(3)]
        labels = ["Alpha", "Beta", "Gamma"]
        html = render_comparison_html(replays, labels)
        for label in labels:
            assert label in html


class TestRenderGamesReport:
    def test_complete_html(self) -> None:
        html = render_games_report([_make_replay()])
        assert "<html>" in html
        assert "</html>" in html
        assert "<body" in html

    def test_contains_title(self) -> None:
        html = render_games_report([_make_replay()], title="My Report")
        assert "My Report" in html

    def test_contains_stats(self) -> None:
        replays = [_make_replay(True), _make_replay(False)]
        html = render_games_report(replays)
        assert "Win rate" in html
        assert "Avg guesses" in html
        assert "50.0%" in html  # 1/2 solved

    def test_default_title(self) -> None:
        html = render_games_report([_make_replay()])
        assert "Evaluation Report" in html
