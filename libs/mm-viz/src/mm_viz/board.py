"""Wordle board HTML renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mm_viz.data import GameReplay

_COLORS: dict[str, str] = {
    "green": "#6aaa64",
    "yellow": "#c9b458",
    "gray": "#787c7e",
    "empty": "#d3d6da",
}


def _tile_html(letter: str, color: str) -> str:
    bg = _COLORS.get(color, _COLORS["empty"])
    text_color = "#ffffff" if color in ("green", "yellow", "gray") else "#000000"
    return (
        f'<div style="width:48px;height:48px;display:flex;align-items:center;'
        f"justify-content:center;background:{bg};color:{text_color};"
        f'font-weight:bold;font-size:24px;border-radius:4px;">'
        f"{letter.upper()}</div>"
    )


def _board_html(replay: GameReplay) -> str:
    word_len = len(replay.target)
    rows: list[str] = []
    for guess, fb in zip(replay.guesses, replay.feedback, strict=True):
        tiles = "".join(_tile_html(guess[i], fb[i]) for i in range(word_len))
        rows.append(f'<div style="display:grid;grid-template-columns:repeat({word_len},48px);gap:4px;">{tiles}</div>')
    return '<div style="display:flex;flex-direction:column;gap:4px;padding:8px;">' + "\n".join(rows) + "</div>"


def render_game_html(replay: GameReplay) -> str:
    """Single game as HTML with colored tiles."""
    return _board_html(replay)


def render_comparison_html(replays: list[GameReplay], labels: list[str]) -> str:
    """Multiple games side-by-side with labels."""
    boards: list[str] = []
    for replay, label in zip(replays, labels, strict=True):
        board = _board_html(replay)
        boards.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;">'
            f'<div style="font-weight:bold;margin-bottom:4px;">{label}</div>'
            f"{board}</div>"
        )
    return '<div style="display:flex;gap:24px;flex-wrap:wrap;">' + "\n".join(boards) + "</div>"


def render_games_report(replays: list[GameReplay], title: str = "Evaluation Report") -> str:
    """Full HTML page with games and summary stats."""
    total = len(replays)
    wins = sum(1 for r in replays if r.solved)
    win_rate = wins / total if total else 0.0
    avg_guesses = sum(r.turns for r in replays) / total if total else 0.0

    boards = "\n".join(
        f'<div style="margin:12px 0;">'
        f'<div style="font-weight:bold;">Game {i + 1} — target: {r.target}'
        f" {'(solved)' if r.solved else '(failed)'}</div>"
        f"{_board_html(r)}</div>"
        for i, r in enumerate(replays)
    )

    return f"""<html>
<head><title>{title}</title></head>
<body style="font-family:sans-serif;padding:24px;">
<h1>{title}</h1>
<div style="margin-bottom:16px;">
  <strong>Win rate:</strong> {win_rate:.1%} ({wins}/{total})<br>
  <strong>Avg guesses:</strong> {avg_guesses:.2f}
</div>
{boards}
</body>
</html>"""
